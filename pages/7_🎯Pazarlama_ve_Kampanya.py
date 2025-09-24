# pages/7_Pazarlama_ve_Kampanya.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, churn_tahmin_modeli_olustur, 
                           clv_hesapla, kampanya_onerileri_uret, kampanya_roi_simulasyonu_yap,
                           optimal_indirim_hesapla,
                           roi_simulasyon_raporu_pdf_olustur, optimal_indirim_raporu_pdf_olustur)

st.set_page_config(page_title="Pazarlama Modülü", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri_guncellenmis.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    return temiz_df, clv_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

st.title("🎯 Pazarlama ve Kampanya Modülü")

st.markdown("---")
st.subheader("Analiz Dönemini Seçin")
min_tarih_pazarlama = temiz_df['Tarih'].min().date()
max_tarih_pazarlama = temiz_df['Tarih'].max().date()
col_tarih_p1, col_tarih_p2 = st.columns(2)
with col_tarih_p1:
    baslangic_tarihi_paz = st.date_input("Başlangıç Tarihi", min_tarih_pazarlama, key="paz_start")
with col_tarih_p2:
    bitis_tarihi_paz = st.date_input("Bitiş Tarihi", max_tarih_pazarlama, key="paz_end")

# Aktif müşterilere göre ana sonuçları filtrele
aktif_musteriler = temiz_df[
    (temiz_df['Tarih'].dt.date >= baslangic_tarihi_paz) & 
    (temiz_df['Tarih'].dt.date <= bitis_tarihi_paz)
]['MusteriID'].unique()
sonuclar_df_filtrelenmis = sonuclar_df[sonuclar_df.index.isin(aktif_musteriler)]

# Sekme sayısı 3'e düşürüldü
tab1, tab2, tab3 = st.tabs(["Kampanya Fikirleri", "ROI Simülasyonu", "Optimal İndirim Analizi"])

with tab1:
    st.header("Segment Bazlı Kampanya Stratejileri")
    st.markdown("Mevcut müşteri segmentlerinize dayanarak hedefe yönelik, aksiyona dönüştürülebilir pazarlama kampanyası fikirleri.")
    kampanya_onerileri = kampanya_onerileri_uret(sonuclar_df_filtrelenmis)
    if kampanya_onerileri:
        for segment, detaylar in kampanya_onerileri.items():
            if detaylar.get('Hedef Kitle Büyüklüğü', 0) > 0:
                with st.expander(f"**{segment}** için Stratejiler", expanded=(segment in ['Şampiyonlar', 'Riskli Müşteriler'])):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"🎯 Hedef: {detaylar['Hedef']}")
                        st.markdown("**💡 Kampanya Fikirleri:**")
                        for fikir in detaylar['Kampanya Fikirleri']: st.markdown(f"- {fikir}")
                    with col2:
                        st.metric("Hedef Kitle Büyüklüğü", f"{detaylar['Hedef Kitle Büyüklüğü']} Müşteri")
                        st.markdown("Detaylı listeyi **'Genel Bakış'** sayfasından görebilirsiniz.")

with tab2:
    st.header("📈 Kampanya ROI Simülatörü")
    st.markdown("Bir kampanya düzenlemeden önce, potansiyel etkisini ve yatırım getirisini (ROI) burada simüle edebilirsiniz.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Hedef Kitleyi Seçin")
        segment_listesi = sonuclar_df_filtrelenmis['Segment'].unique().tolist()
        if not segment_listesi:
            st.warning("Analiz edilecek segment bulunamadı.")
        else:
            hedef_segment = st.selectbox("Hangi segmente kampanya yapmak istersiniz?", segment_listesi)

    with col2:
        st.subheader("2. Kampanya Parametrelerini Girin")
        musteri_basi_maliyet = st.number_input("Müşteri Başına İletişim Maliyeti (€)", min_value=0.0, value=0.1, step=0.01)
        indirim_orani = st.slider("Uygulanacak İndirim Oranı (%)", 0, 50, 10)
    
    st.subheader("3. Kampanya Etkisini Varsayın")
    beklenen_etki_orani = st.slider(
        "Kampanyanın, müşterinin satın alma olasılığını ne kadar artırmasını bekliyorsunuz? (%)",
        0, 100, 20,
        help="Örn: %20 seçmek, normalde %50 ihtimalle alım yapacak bir müşterinin olasılığını %60'a çıkarır (%50 * 1.20)."
    )
    
    if 'hedef_segment' in locals():
        if st.button("ROI Simülasyonunu Çalıştır", type="primary"):
            with st.spinner("Simülasyon hesaplanıyor..."):
                simulasyon_sonuclari = kampanya_roi_simulasyonu_yap(sonuclar_df_filtrelenmis, hedef_segment, beklenen_etki_orani, indirim_orani, musteri_basi_maliyet)
            
            st.session_state.simulasyon_sonuclari = simulasyon_sonuclari
            st.session_state.simulasyon_parametreleri = (hedef_segment, musteri_basi_maliyet, indirim_orani, beklenen_etki_orani)

    if 'simulasyon_sonuclari' in st.session_state:
        simulasyon_sonuclari = st.session_state.simulasyon_sonuclari
        hedef_segment, musteri_basi_maliyet, indirim_orani, beklenen_etki_orani = st.session_state.simulasyon_parametreleri

        st.markdown("---"); st.subheader("📊 Simülasyon Sonuçları")
        col1_res, col2_res, col3_res = st.columns(3)
        col1_res.metric("Hedeflenen Müşteri Sayısı", f"{simulasyon_sonuclari['Hedef Kitle Sayısı']:.0f}")
        col2_res.metric("Tahmini Ekstra Müşteri", f"{simulasyon_sonuclari['Tahmini Ekstra Müşteri']:.1f}")
        col3_res.metric("Tahmini Toplam Ciro", f"{simulasyon_sonuclari['Tahmini Toplam Ciro']:,.0f} €")
        
        col4_res, col5_res, col6_res = st.columns(3)
        col4_res.metric("Toplam Kampanya Maliyeti", f"{simulasyon_sonuclari['Toplam Maliyet']:,.0f} €", delta_color="inverse")
        col5_res.metric("Tahmini Net Kar", f"{simulasyon_sonuclari['Tahmini Net Kar']:,.0f} €")
        col6_res.metric("Tahmini ROI", f"{simulasyon_sonuclari['Tahmini ROI (%)']:.1f}%")
        st.markdown("---")

        pdf_bytes = roi_simulasyon_raporu_pdf_olustur(simulasyon_sonuclari, hedef_segment, musteri_basi_maliyet, indirim_orani, beklenen_etki_orani)
        st.download_button(label="📄 Bu Simülasyon Raporunu İndir (.pdf)", data=pdf_bytes, file_name=f"roi_sim_{hedef_segment}.pdf", mime="application/pdf")

with tab3: # Eskiden tab4 olan bölüm artık tab3
    st.header("💸 Optimal İndirim Oranı Analizi")
    st.markdown("Bu araç, seçtiğiniz hedef ve önceliklere göre net kar ve müşteri kazanım etkisini dengeleyerek en uygun indirim oranını önerir.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Hedef ve Maliyetler")
        segment_listesi_opt = sonuclar_df_filtrelenmis['Segment'].unique().tolist()
        hedef_segment_opt = st.selectbox("Optimizasyon için hedef segmenti seçin:", segment_listesi_opt, key="opt_segment")
        musteri_basi_maliyet_opt = st.number_input("Müşteri Başına İletişim Maliyeti (€)", min_value=0.0, value=0.1, step=0.01, key="opt_maliyet")

    with col2:
        st.subheader("2. Stratejik Öncelik")
        agirlik_kar = st.slider("Kar Odaklılık Ağırlığı (%)", 0, 100, 70, help="Stratejiniz ne kadar kar odaklı? %100 seçerseniz sadece karı maksimize eder.")
        agirlik_etki = 100 - agirlik_kar
        st.write(f"Öncelikler: **%{agirlik_kar} Kar** vs. **%{agirlik_etki} Etki**")

    if st.button("Optimal İndirim Oranını Hesapla", type="primary"):
        if 'hedef_segment_opt' in locals() and hedef_segment_opt:
            with st.spinner("Optimizasyon yapılıyor..."):
                optimizasyon_df, optimal_nokta = optimal_indirim_hesapla(sonuclar_df_filtrelenmis,
                    hedef_segment_opt, musteri_basi_maliyet_opt,
                    agirlik_kar=agirlik_kar/100, agirlik_etki=agirlik_etki/100
                )
            
            st.session_state.optimizasyon_df = optimizasyon_df
            st.session_state.optimal_nokta = optimal_nokta

    if 'optimal_nokta' in st.session_state and st.session_state.optimal_nokta is not None:
        optimal_nokta = st.session_state.optimal_nokta
        optimizasyon_df = st.session_state.optimizasyon_df
        st.subheader("Optimizasyon Sonuçları")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("Optimal İndirim Oranı", f"%{optimal_nokta['İndirim Oranı (%)']:.0f}")
            st.metric("Bu Orandaki Tahmini Net Kar", f"{optimal_nokta['Tahmini Net Kar (€)']:,.0f} €")
            st.metric("Bu Orandaki Beklenen Etki", f"%{optimal_nokta['Beklenen Etki (%)']:.1f}")
        with col_res2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=optimizasyon_df["İndirim Oranı (%)"], y=optimizasyon_df["Tahmini Net Kar (€)"], name="Tahmini Net Kar (€)", line=dict(color='#6c63ff')))
            fig.add_trace(go.Scatter(x=optimizasyon_df["İndirim Oranı (%)"], y=optimizasyon_df["Beklenen Etki (%)"], name="Beklenen Etki (%)", yaxis="y2", line=dict(color='#ff6347', dash='dot')))
            fig.add_vline(x=optimal_nokta['İndirim Oranı (%)'], line_dash="dash", line_color="red", annotation_text="Optimal Nokta")
            fig.update_layout(title=f"'{hedef_segment_opt}' İçin Kar ve Etki Optimizasyonu", xaxis_title="İndirim Oranı (%)",
                              yaxis=dict(title="Tahmini Net Kar (€)"), yaxis2=dict(title="Beklenen Etki (%)", overlaying="y", side="right"),
                              legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)

        pdf_bytes = optimal_indirim_raporu_pdf_olustur(optimal_nokta, fig)
        st.download_button(label="📄 Bu Optimizasyon Raporunu İndir (.pdf)", data=pdf_bytes, file_name=f"opt_indirim_{hedef_segment_opt}.pdf", mime="application/pdf")
