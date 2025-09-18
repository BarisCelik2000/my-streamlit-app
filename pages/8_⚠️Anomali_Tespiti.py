# pages/8_Anomali_Tespiti.py
import streamlit as st
import pandas as pd
import plotly.express as px
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           anomali_tespiti_yap, anomali_tespiti_dbscan, 
                           davranissal_anomali_tespiti_yap, anomali_gruplama_yap, 
                           islem_bazli_anomali_tespiti_yap, 
                           anomali_nedenlerini_acikla)

st.set_page_config(page_title="Anomali Tespiti", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    return temiz_df, clv_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

st.title("⚠️ Anomali (Aykırı Değer) Tespiti")



tab1, tab2, tab3 = st.tabs(["Genel Profil Anomalileri", "Davranışsal Anomaliler (Erken Uyarı)", "İşlem Bazlı Anomaliler"])

with tab1:
    st.header("Genel Profil Anomalileri (Tüm Müşterilere Kıyasla)")
    st.markdown("Bu analiz, bir müşterinin RFM profilinin, **genel müşteri kitlesinin** normal davranış kalıplarından ne kadar saptığını gösterir.")
    
    algoritma_secimi = st.radio("Kullanılacak Algoritmayı Seçin:", ('Isolation Forest', 'DBSCAN'), horizontal=True, key="algo_radio")
    
    st.markdown("---")
    st.header("Analiz Parametreleri")

    # Isolation Forest algoritması seçildiğinde...
    if algoritma_secimi == 'Isolation Forest':
        kontaminasyon = st.slider("Tahmini Anomali Oranı (%)", 1, 20, 5, 1, help="Veri setinizde ne kadar oranda (% olarak) anomali beklediğinizi belirtir.", key="slider_profil_iso") / 100.0
        if st.button("Isolation Forest ile Anomalileri Tespit Et", type="primary"):
            with st.spinner("Anomali tespiti yapılıyor..."):
                anomali_sonuclari_df = anomali_tespiti_yap(sonuclar_df.copy(), kontaminasyon_orani=kontaminasyon)
                st.session_state.anomali_sonuclari_df = anomali_sonuclari_df
    
    # DBSCAN algoritması seçildiğinde...
    elif algoritma_secimi == 'DBSCAN':
        col1, col2 = st.columns(2)
        with col1:
            eps_degeri = st.slider("Eps Değeri (Komşuluk Mesafesi)", 0.1, 2.0, 0.5, 0.1)
        with col2:
            min_samples_degeri = st.slider("Min Samples Değeri (Küme Yoğunluğu)", 2, 20, 5, 1)
        if st.button("DBSCAN ile Anomalileri Tespit Et", type="primary"):
            with st.spinner("Yoğunluk bazlı anomali tespiti yapılıyor..."):
                anomali_sonuclari_df = anomali_tespiti_dbscan(sonuclar_df.copy(), eps=eps_degeri, min_samples=min_samples_degeri)
                st.session_state.anomali_sonuclari_df = anomali_sonuclari_df

    # Analiz sonuçları varsa göster
    if 'anomali_sonuclari_df' in st.session_state and not st.session_state.anomali_sonuclari_df.empty:
        anomali_sonuclari_df = st.session_state.anomali_sonuclari_df
        st.success("Analiz tamamlandı!")
        st.markdown("---")
        
        anormal_musteriler = anomali_sonuclari_df[anomali_sonuclari_df['Anomali_Etiketi'] == -1].copy()
        
        if anormal_musteriler.empty:
            st.info("Belirtilen parametrelerle herhangi bir anomali tespit edilmedi.")
        else:
            with st.spinner("Anomali nedenleri analiz ediliyor..."):
                nedenler_sozlugu = anomali_nedenlerini_acikla(anomali_sonuclari_df)
                anormal_musteriler['Anomali Nedeni'] = anormal_musteriler.index.map(nedenler_sozlugu)
            
            st.subheader("Tespit Edilen Anormal Müşteriler")
            st.warning(f"Toplam **{len(anormal_musteriler)}** müşteri, normal davranış kalıplarının dışında olarak etiketlendi.")
            
            # --- GÜNCELLENMİŞ TABLO GÖSTERİMİ ---
            gosterilecek_sutunlar = ['Anomali Nedeni', 'Segment', 'Recency', 'Frequency', 'Monetary']
            
            # Sadece Isolation Forest skor ürettiği için, skor sütununu ve stili ona göre ekle
            if 'Anomali_Skoru' in anormal_musteriler.columns:
                gosterilecek_sutunlar.append('Anomali_Skoru')
                st.info("Tablo, en aykırı müşterileri (skoru en düşük olanlar) en üstte gösterecek şekilde sıralanmıştır.")
                st.dataframe(anormal_musteriler[gosterilecek_sutunlar]
                             .sort_values('Anomali_Skoru', ascending=True)
                             .style.background_gradient(cmap='Reds_r', subset=['Anomali_Skoru'])
                             .format({'Anomali_Skoru': '{:.2f}'}))
            else: # DBSCAN için
                st.dataframe(anormal_musteriler[gosterilecek_sutunlar])
            
            st.markdown("---")
            st.subheader("Anomali Grupları (Clustering)")
            st.markdown("Tespit edilen anormal müşterileri, davranışsal profillerine göre gruplayarak daha derinlemesine analiz edin.")
            
            kume_sayisi = st.slider("Oluşturulacak Anomali Grubu Sayısı", 2, 5, 3, 1)
            
            if st.button("Anomalileri Grupla", type="primary"):
                with st.spinner("Anormal müşteriler gruplanıyor..."):
                    gruplanmis_anomaliler, merkezler = anomali_gruplama_yap(anormal_musteriler.copy(), kume_sayisi=kume_sayisi)
                
                if not merkezler.empty:
                    st.success("Gruplama tamamlandı!")
                    st.subheader("Grup Profilleri (Ortalama RFM Değerleri)")
                    st.dataframe(merkezler.style.format("{:.0f}"))
                    
                    grup_tab_listesi = st.tabs(list(merkezler.index))
                    for i, grup_adi in enumerate(merkezler.index):
                        with grup_tab_listesi[i]:
                            st.subheader(f"'{grup_adi}' Grubundaki Müşteriler")
                            st.dataframe(gruplanmis_anomaliler[gruplanmis_anomaliler['Anomali_Grubu'] == grup_adi][['Segment', 'Recency', 'Frequency', 'Monetary']])
                else:
                    st.error("Gruplama için yeterli sayıda anormal müşteri bulunamadı.")
            
            st.markdown("---")
            st.subheader("Anomalilerin 3 Boyutlu Görselleştirmesi")
            plot_df = anomali_sonuclari_df.copy()
            plot_df['Anomali_Durumu'] = plot_df['Anomali_Etiketi'].apply(lambda x: 'Anomali' if x == -1 else 'Normal')
            fig = px.scatter_3d(
                plot_df, x='Recency', y='Frequency', z='Monetary', color='Anomali_Durumu',
                color_discrete_map={'Anomali': 'red', 'Normal': 'blue'},
                hover_data=[plot_df.index]
            )
            st.plotly_chart(fig, use_container_width=True)

# --- SEKME 2: DAVRANIŞSAL ANOMALİLER ---
with tab2:
    st.header("Davranışsal Anomaliler (Müşterinin Kendi Alışkanlığına Kıyasla)")
    st.markdown("Bu erken uyarı sistemi, bir müşterinin **kendi normal satın alma ritminin dışına çıktığı** anları tespit eder.")
    
    hassasiyet = st.slider(
        "Hassasiyet Eşiği (Standart Sapma)",
        min_value=1.5, max_value=4.0, value=2.5, step=0.1,
        help="Bir satın alma aralığının 'anormal' sayılması için, müşterinin kendi ortalamasından kaç standart sapma daha uzun olması gerektiğini belirtir.",
        key="slider_davranis"
    )

    if st.button("Davranışsal Anomalileri Tespit Et", type="primary"):
        with st.spinner("Müşterilerin alışkanlıkları analiz ediliyor..."):
            davranissal_anomaliler_df = davranissal_anomali_tespiti_yap(temiz_df, hassasiyet=hassasiyet)
            st.session_state.davranissal_anomaliler_df = davranissal_anomaliler_df
            
    if 'davranissal_anomaliler_df' in st.session_state:
        davranissal_anomaliler_df = st.session_state.davranissal_anomaliler_df
        st.success("Analiz tamamlandı!")
        st.subheader("Alışkanlıklarını Bozan Müşteriler (Erken Uyarı Listesi)")
        
        if davranissal_anomaliler_df.empty:
            st.info("Belirtilen hassasiyet seviyesinde, kendi satın alma ritmini bozan müşteri tespit edilmedi.")
        else:
            st.warning(f"Toplam **{len(davranissal_anomaliler_df)}** müşterinin son satın alma davranışında bir anomali tespit edildi.")
            st.dataframe(davranissal_anomaliler_df[[
                'MusteriID', 'Son_Alim_Tarihi', 'Gecen_Sure_Gun', 'Ortalama_Gun', 'Son_Alim_Araligi'
            ]].rename(columns={
                'MusteriID': 'Müşteri ID','Son_Alim_Tarihi': 'Anomalinin Tespit Edildiği Alım','Gecen_Sure_Gun': 'O Günden Bugüne Geçen Süre',
                'Ortalama_Gun': 'Normal Alım Aralığı (Ort. Gün)','Son_Alim_Araligi': 'Anormal Alım Aralığı (Gün)'
            }).style.format({
                'Anomalinin Tespit Edildiği Alım': lambda x: x.strftime('%d-%m-%Y'), 'O Günden Bugüne Geçen Süre': '{:.0f}',
                'Normal Alım Aralığı (Ort. Gün)': '{:.1f}', 'Anormal Alım Aralığı (Gün)': '{:.0f}'
            }))

            # --- YENİ EKLENEN BÖLÜM: Davranışsal Anomali Trend Grafiği ---
            st.markdown("---")
            st.subheader("Davranışsal Anomali Trendi")
            st.markdown("Tespit edilen anomalilerin aylara göre dağılımı.")

            df_trend = davranissal_anomaliler_df.copy()
            # Anomaliyi tespit ettiğimiz son alım tarihine göre aylık gruplama yap
            df_trend['Anomali_Ayi'] = pd.to_datetime(df_trend['Son_Alim_Tarihi']).dt.to_period('M')
            
            anomali_sayilari_aylik = df_trend.groupby('Anomali_Ayi').size().reset_index(name='Anomali_Sayisi')
            anomali_sayilari_aylik['Anomali_Ayi'] = anomali_sayilari_aylik['Anomali_Ayi'].dt.to_timestamp()
            
            fig_trend = px.bar(
                anomali_sayilari_aylik,
                x='Anomali_Ayi',
                y='Anomali_Sayisi',
                title='Aylara Göre Tespit Edilen Davranışsal Anomali Sayısı',
                labels={'Anomali_Ayi': 'Ay', 'Anomali_Sayisi': 'Anomali Sayısı'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# --- SEKME 3: İŞLEM BAZLI ANOMALİLER ---
with tab3:
    st.header("İşlem Bazlı Anomaliler (Sahtekarlık/Fırsat Tespiti)")
    st.markdown("Bu analiz, **tekil işlem bazında** aykırı durumları tespit eder.")
    
    kontaminasyon_islem = st.slider(
        "Tahmini Anormal İşlem Oranı (%)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Tüm işlemlerin yüzde kaçının anormal olmasını beklediğinizi belirtir.",
        key="slider_islem"
    ) / 100.0

    if st.button("İşlem Anomalilerini Tespit Et", type="primary"):
        with st.spinner("Tüm işlemler analiz ediliyor..."):
            islem_anomalileri_df = islem_bazli_anomali_tespiti_yap(temiz_df, kontaminasyon_orani=kontaminasyon_islem)
            st.session_state.islem_anomalileri_df = islem_anomalileri_df # Sonuçları kaydet
            
    if 'islem_anomalileri_df' in st.session_state:
        islem_anomalileri_df = st.session_state.islem_anomalileri_df
        st.success("Analiz tamamlandı!")
        st.subheader("Tespit Edilen Anormal İşlemler")
        
        if islem_anomalileri_df.empty:
            st.info("Belirtilen hassasiyet seviyesinde herhangi bir anormal işlem tespit edilmedi.")
        else:
            st.warning(f"Toplam **{len(islem_anomalileri_df)}** adet anormal işlem tespit edildi.")
            st.info("Tablo, en aykırı işlemleri (skoru en düşük olanlar) en üstte gösterecek şekilde sıralanmıştır.")
            
            # --- GÜNCELLENMİŞ TABLO GÖSTERİMİ ---
            st.dataframe(islem_anomalileri_df[[
                'MusteriID', 'Tarih', 'UrunKodu', 'Miktar', 'BirimFiyat', 'ToplamTutar', 'Anomali_Skoru'
            ]].sort_values('Anomali_Skoru', ascending=True)
              .style.format({
                'BirimFiyat': '{:,.2f} €', 'ToplamTutar': '{:,.2f} €', 'Anomali_Skoru': '{:.3f}'
            }).background_gradient(cmap='Reds_r', subset=['Anomali_Skoru']))
            
            st.subheader("Anormal İşlemlerin Görselleştirilmesi")
            fig = px.scatter(
                islem_anomalileri_df, x='Miktar', y='ToplamTutar', color='Anomali_Skoru',
                color_continuous_scale=px.colors.sequential.Reds_r,
                hover_data=['MusteriID', 'UrunKodu', 'Tarih'],
                title="Anormal İşlemlerin Miktar ve Tutar Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)