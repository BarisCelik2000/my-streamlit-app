# pages/12_Mevsimsellik_ve_Trendler.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_handler import veriyi_yukle_ve_temizle, genel_satis_trendi_hazirla
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           zaman_serisi_ayristirma_yap, gelecek_tahmini_yap, trend_analizi_yap,
                           mevsimsellik_analizi_yap)

st.set_page_config(page_title="Mevsimsellik ve Trend Analizi", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.xlsx' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    return temiz_df, clv_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()


st.title("📈 Mevsimsellik ve Trend Analizi")
st.markdown("Bu sayfa, satış verilerinizdeki uzun vadeli trendleri, mevsimsel döngüleri ve gelecek tahminlerini analiz eder.")


st.markdown("---")
st.header("Analiz Kapsamını Seçin")

segment_listesi = ['Tüm Müşteriler'] + sonuclar_df['Segment'].unique().tolist()
secilen_segment = st.selectbox("Hangi müşteri grubunu analiz etmek istersiniz?", segment_listesi)

if secilen_segment == 'Tüm Müşteriler':
    analiz_df = temiz_df
    baslik_eki = " (Tüm Müşteriler)"
else:
    hedef_musteri_idler = sonuclar_df[sonuclar_df['Segment'] == secilen_segment].index
    analiz_df = temiz_df[temiz_df['MusteriID'].isin(hedef_musteri_idler)]
    baslik_eki = f" ({secilen_segment})"

aylik_veri = genel_satis_trendi_hazirla(analiz_df)

st.markdown("---")

# Sekme sayısı 4'e çıkarıldı ve yeni sekme en başa eklendi
tab1, tab2, tab3, tab4 = st.tabs(["Yıllık/Aylık Performans", "Zaman Serisi Ayrıştırma", "Tatil Etkisi Analizi", "🔮 Gelecek Tahmini"])

# --- YENİ EKLENEN SEKME 1: Yıllık/Aylık Performans Isı Haritası ---
with tab1:
    st.header("Yıllık ve Aylık Performans Isı Haritası" + baslik_eki)
    st.markdown("Bu ısı haritası, farklı yıllardaki aylık ciro performansınızı karşılaştırmanızı sağlar.")

    if analiz_df.empty:
        st.warning("Bu segment için görüntülenecek veri bulunamadı.")
    else:
        df_heatmap = analiz_df.copy()
        df_heatmap['Yıl'] = df_heatmap['Tarih'].dt.year
        df_heatmap['Ay'] = df_heatmap['Tarih'].dt.month
        
        performans_pivot = df_heatmap.pivot_table(
            values='ToplamTutar', 
            index='Yıl', 
            columns='Ay', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Sütunları ay isimleriyle değiştirme ve sıralama
        ay_isimleri = {
            1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
            7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
        }
        performans_pivot.rename(columns=ay_isimleri, inplace=True)
        
        # Eksik ay sütunlarını 0 ile doldur
        for i in range(1, 13):
            if ay_isimleri[i] not in performans_pivot.columns:
                performans_pivot[ay_isimleri[i]] = 0
        
        # Ay sırasının doğru olması için yeniden sırala
        performans_pivot = performans_pivot[list(ay_isimleri.values())]

        fig_heatmap = px.imshow(
            performans_pivot,
            text_auto=',.0f',
            aspect="auto",
            labels=dict(x="Aylar", y="Yıllar", color="Toplam Ciro (€)"),
            title=f"Yıllık ve Aylık Ciro Performansı{baslik_eki}",
            color_continuous_scale=px.colors.sequential.Greens
        )
        fig_heatmap.update_xaxes(side="top")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with st.expander("📊 Isı Haritası Nasıl Yorumlanır?"):
            st.info("""
            - **Dikey Karşılaştırma:** Bir aydaki hücreleri yukarıdan aşağıya doğru karşılaştırarak o ayın **yıllar arası büyüme** performansını görebilirsiniz. (Örn: 2023'ün Mart ayı, 2022'nin Mart ayından daha mı koyu renkli?)
            - **Yatay Karşılaştırma:** Bir yıldaki hücreleri soldan sağa doğru takip ederek o yılın içindeki **mevsimsel trendleri** görebilirsiniz. (Örn: Hangi aylar genellikle daha koyu/açık renkli?)
            """)

with tab2:
    st.header("Zaman Serisi Bileşenlerine Ayırma" + baslik_eki)
    
    model_tipi = st.selectbox(
        "Ayrıştırma Modelini Seçin:",
        ('additive', 'multiplicative'),
        help="Eğer mevsimsel dalgalanmalar zamanla sabit kalıyorsa 'additive', ciro arttıkça dalgalanma da artıyorsa 'multiplicative' seçilir."
    )

    if not aylik_veri.empty and len(aylik_veri) >= 24:
        with st.spinner(f"'{secilen_segment}' için zaman serisi bileşenlerine ayrıştırılıyor..."):
            ayristirma, hata_mesaji = zaman_serisi_ayristirma_yap(aylik_veri, model_tipi=model_tipi)
        
        if hata_mesaji:
            st.error(hata_mesaji)
        elif ayristirma:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=("Gözlemlenen Veri", "Trend", "Mevsimsellik", "Kalıntılar"))

            fig.add_trace(go.Scatter(x=ayristirma.observed.index, y=ayristirma.observed, name='Gözlemlenen'), row=1, col=1)
            fig.add_trace(go.Scatter(x=ayristirma.trend.index, y=ayristirma.trend, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=ayristirma.seasonal.index, y=ayristirma.seasonal, name='Mevsimsellik'), row=3, col=1)
            fig.add_trace(go.Scatter(x=ayristirma.resid.index, y=ayristirma.resid, name='Kalıntı', mode='markers'), row=4, col=1)
            
            fig.update_layout(height=800, title_text="Satış Cirosunun Bileşenleri" + baslik_eki)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 Trend Analizi Sonuçları")
            
            trend_sonuclari = trend_analizi_yap(ayristirma.trend)
            
            if trend_sonuclari:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Genel Trend Yönü", trend_sonuclari['yon'])
                with col2:
                    st.metric("Ortalama Aylık Büyüme/Küçülme", f"{trend_sonuclari['egim']:,.0f} €")
                st.info("Ortalama aylık büyüme/küçülme, trend çizgisinin eğimine dayalı bir tahmindir ve cironuzun her ay ortalama ne kadar arttığını veya azaldığını gösterir.")
            else:
                st.warning("Trend analizi için yeterli veri bulunamadı.")

            st.markdown("---")
            st.subheader("🗓️ Mevsimsel Etki Analizi")
            mevsimsellik_df = mevsimsellik_analizi_yap(ayristirma.seasonal)
            if not mevsimsellik_df.empty:
                fig_bar = px.bar(
                    mevsimsellik_df,
                    x='Ay',
                    y='etki',
                    title="Ayların Ortalama Mevsimsel Etkisi",
                    labels={'Ay': 'Ay', 'etki': 'Ortalama Etki (€)'},
                    color='etki',
                    color_continuous_scale='RdBu'
                )
                fig_bar.add_hline(y=0)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.info("Bu grafik, her bir ayın ciroya olan ortalama pozitif (mavi) veya negatif (kırmızı) etkisini gösterir. Pazarlama ve stok planlaması için kullanılabilir.")

    else:
        st.warning("Bu analiz için seçilen grupta en az 24 aylık (2 yıl) satış verisi gerekmektedir.")


with tab3:
    st.header("📅 Tatil ve Özel Gün Etkisi Analizi" + baslik_eki)
    st.markdown("Türkiye'deki resmi tatillerin seçtiğiniz müşteri grubunun satışları üzerindeki etkisini analiz edin.")
    
    if st.button("Tatil Etkisini Analiz Et", type="primary"):
        with st.spinner(f"'{secilen_segment}' için Prophet modeli eğitiliyor..."):
            model, forecast = gelecek_tahmini_yap(aylik_veri)
            st.session_state[f'prophet_model_{secilen_segment}'] = model
            st.session_state[f'prophet_forecast_{secilen_segment}'] = forecast
            
    session_key_model = f'prophet_model_{secilen_segment}'
    if session_key_model in st.session_state:
        st.success("Analiz tamamlandı!")
        
        model = st.session_state[session_key_model]
        forecast = st.session_state[f'prophet_forecast_{secilen_segment}']
        
        st.subheader("Modelin Bileşenleri")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
        
        st.subheader("Önemli Tatillerin Etkisi (Sayısal)")
        holidays_df = forecast[forecast['holidays'] != 0][['ds', 'holidays']]
        holidays_df['Tatil Etkisi'] = holidays_df['holidays'].apply(lambda x: f"{x:,.0f} €")
        holidays_df['ds'] = holidays_df['ds'].dt.date
        st.dataframe(holidays_df[['ds', 'Tatil Etkisi']].rename(columns={'ds':'Tarih'}))

with tab4:
    st.header("🔮 Gelecek Dönem Satış Tahmini" + baslik_eki)
    st.markdown("Prophet modelini kullanarak, seçtiğiniz müşteri grubunun gelecek 12 aydaki potansiyel satış trendini öngörün.")
    
    if not aylik_veri.empty and len(aylik_veri) >= 12:
        if st.button("Gelecek 12 Ay İçin Tahmin Yap", type="primary"):
            with st.spinner(f"'{secilen_segment}' için gelecek tahmini yapılıyor..."):
                model, forecast = gelecek_tahmini_yap(aylik_veri, tahmin_periyodu_ay=12)
                st.session_state[f'forecast_model_{secilen_segment}'] = model
                st.session_state[f'forecast_data_{secilen_segment}'] = forecast

        session_key = f'forecast_model_{secilen_segment}'
        if session_key in st.session_state:
            st.success("Tahmin tamamlandı!")
            
            model = st.session_state[session_key]
            forecast = st.session_state[f'forecast_data_{secilen_segment}']
            
            st.subheader("Satış Tahmin Grafiği")
            fig = model.plot(forecast)
            ax = fig.gca()
            ax.set_title("Gelecek 12 Aylık Satış Tahmini" + baslik_eki)
            ax.set_xlabel("Tarih")
            ax.set_ylabel("Tahmini Ciro (€)")
            st.pyplot(fig)
            
            with st.expander("Tahmin Verilerini Detaylı Görüntüle"):
                gelecek_tahminleri = forecast[forecast['ds'] > aylik_veri['ds'].max()]
                format_sozlugu = {
                    'Tahmin': '{:,.0f} €',
                    'Alt Sınır': '{:,.0f} €',
                    'Üst Sınır': '{:,.0f} €'
                }
                
                st.dataframe(gelecek_tahminleri[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Tarih', 'yhat': 'Tahmin', 'yhat_lower': 'Alt Sınır', 'yhat_upper': 'Üst Sınır'
                }).style.format(formatter=format_sozlugu))
    else:
        st.warning("Bu analiz için seçilen grupta en az 12 aylık satış verisi gerekmektedir.")
