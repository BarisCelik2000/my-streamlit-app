# pages/2_Müşteri_Detayı.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle, musteri_zaman_serisi_hazirla
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla, 
                           market_basket_analizi_yap, satis_tahmini_yap, 
                           tahmin_grafigini_ciz, urun_tavsiyesi_uret,
                           pdf_raporu_olustur, musteri_yolculugu_analizi_yap)

st.set_page_config(page_title="Müşteri Detayı", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri_guncellenmis.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    
    birliktelik_kurallari = market_basket_analizi_yap(temiz_df)
    yolculuk_pivot, _, _, _ = musteri_yolculugu_analizi_yap(temiz_df, clv_df)
    
    return temiz_df, clv_df, birliktelik_kurallari, yolculuk_pivot

st.title("👤 Müşteri Detay Analizi ve Satış Tahmini")

temiz_df, sonuclar_df, birliktelik_kurallari, yolculuk_pivot = veriyi_getir_ve_isle()

musteri_listesi = sonuclar_df.index.tolist()
secilen_musteri = st.selectbox("Analiz Yapmak İçin Müşteri Seçin", musteri_listesi)

if secilen_musteri:
    if secilen_musteri in st.session_state.get('profil_anomalileri', []):
        st.warning(f"**Profil Anomalisi:** Bu müşteri, genel müşteri profillerine göre aykırı bir RFM skoruna sahiptir.")
    if secilen_musteri in st.session_state.get('davranissal_anomaliler', []):
        st.error(f"**Davranışsal Anomali Uyarısı:** Bu müşteri, kendi normal satın alma ritmini bozmuştur. Churn riski artmış olabilir!")    
    
    st.markdown("---")
    
    musteri_verisi = sonuclar_df.loc[secilen_musteri]
    musteri_satis_verisi = temiz_df[temiz_df['MusteriID'] == secilen_musteri]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segment", musteri_verisi['Segment'])
    col2.metric("Performans Skoru (MPS)", f"{musteri_verisi['MPS']:.0f}")
    col3.metric("Churn Olasılığı", f"%{musteri_verisi['Churn_Olasiligi']*100:.1f}")
    col4.metric("Yaşam Boyu Değeri (CLV)", f"{musteri_verisi['CLV_Net_Kar']:,.0f} €")
    st.markdown("---")

    # --- YENİ BÖLÜM: Tarih Filtresi ---
    st.subheader("Analiz Dönemini Seçin")
    min_tarih = musteri_satis_verisi['Tarih'].min().date()
    max_tarih = musteri_satis_verisi['Tarih'].max().date()
    col_tarih1, col_tarih2 = st.columns(2)
    with col_tarih1:
        baslangic_tarihi_detay = st.date_input("Başlangıç Tarihi", min_tarih, min_value=min_tarih, max_value=max_tarih, key="detay_start")
    with col_tarih2:
        bitis_tarihi_detay = st.date_input("Bitiş Tarihi", max_tarih, min_value=min_tarih, max_value=max_tarih, key="detay_end")

    musteri_satis_verisi_filtrelenmis = musteri_satis_verisi[
        (musteri_satis_verisi['Tarih'].dt.date >= baslangic_tarihi_detay) & 
        (musteri_satis_verisi['Tarih'].dt.date <= bitis_tarihi_detay)
    ]
    st.markdown("---")
    # --- YENİ BÖLÜM SONU ---
    
    st.subheader("📋 Geçmiş Davranış Özeti")
    col_ozet1, col_ozet2 = st.columns(2)
    
    with col_ozet1:
        st.markdown("**Alışveriş Alışkanlıkları (Seçilen Dönem)**")
        
        if not musteri_satis_verisi_filtrelenmis.empty:
            if 'Kategori' in musteri_satis_verisi_filtrelenmis.columns:
                en_cok_alinan_kategori = musteri_satis_verisi_filtrelenmis['Kategori'].mode()
                if not en_cok_alinan_kategori.empty:
                    st.write(f"🏷️ **Favori Kategorisi:** {en_cok_alinan_kategori.iloc[0]}")

            gunluk_harcama = musteri_satis_verisi_filtrelenmis.groupby(musteri_satis_verisi_filtrelenmis['Tarih'].dt.date)['ToplamTutar'].sum()
            st.write(f"🛒 **Ortalama Sepet Tutarı:** {gunluk_harcama.mean():,.2f} €")
            
            alisveris_gunleri = musteri_satis_verisi_filtrelenmis['Tarih'].dt.date.unique()
            if len(alisveris_gunleri) > 1:
                alisveris_gunleri.sort()
                ortalama_gun_farki = (pd.to_datetime(alisveris_gunleri[1:]) - pd.to_datetime(alisveris_gunleri[:-1])).to_series().dt.days.mean()
                st.write(f"🔄 **Alışveriş Sıklığı (Ort.):** {ortalama_gun_farki:.1f} günde bir")
        else:
            st.info("Seçilen dönemde müşterinin işlemi bulunmuyor.")

    with col_ozet2:
        st.markdown("**Dönemin En Popüler 5 Ürünü (Ciroya Göre)**")
        if not musteri_satis_verisi_filtrelenmis.empty:
            top_5_urunler = musteri_satis_verisi_filtrelenmis.groupby('UrunKodu')['ToplamTutar'].sum().nlargest(5)
            st.dataframe(top_5_urunler.reset_index().style.format({'ToplamTutar': '{:,.2f} €'}))
        else:
            st.info("Seçilen dönemde gösterilecek ürün bulunmuyor.")
    st.markdown("---")

    st.subheader("📊 Müşterinin Kendi Segmentine Göre Konumu")
    musteri_segmenti = musteri_verisi['Segment']
    segment_verisi = sonuclar_df[sonuclar_df['Segment'] == musteri_segmenti]
    segment_ortalamalari = segment_verisi.mean(numeric_only=True)
    col_bench1, col_bench2, col_bench3 = st.columns(3)
    with col_bench1:
        fig = go.Figure(go.Indicator(mode = "gauge+number+delta", value = musteri_verisi['CLV_Net_Kar'], title = {'text': "Yaşam Boyu Değeri (CLV)"}, delta = {'reference': segment_ortalamalari['CLV_Net_Kar'], 'relative': True, 'valueformat': '.0%'}, gauge = {'axis': {'range': [0, segment_verisi['CLV_Net_Kar'].max()]}, 'bar': {'color': "#6c63ff"}, 'steps' : [{'range': [0, segment_ortalamalari['CLV_Net_Kar']], 'color': "lightgray"},], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': segment_ortalamalari['CLV_Net_Kar']}}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50)); st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Kırmızı çizgi, '{musteri_segmenti}' segmentinin ortalama CLV'sini gösterir.")
    with col_bench2:
        fig = go.Figure(go.Indicator(mode = "gauge+number+delta", value = musteri_verisi['Churn_Olasiligi'] * 100, number = {'suffix': "%"}, title = {'text': "Churn Olasılığı (%)"}, delta = {'reference': segment_ortalamalari['Churn_Olasiligi'] * 100, 'relative': False, 'decreasing': {'color': "green"}, 'increasing': {'color': "red"}}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#ff6347"}, 'steps' : [{'range': [0, segment_ortalamalari['Churn_Olasiligi'] * 100], 'color': "lightgray"},], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': segment_ortalamalari['Churn_Olasiligi'] * 100}}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50)); st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Kırmızı çizgi, '{musteri_segmenti}' segmentinin ortalama Churn olasılığını gösterir.")
    with col_bench3:
        fig = go.Figure(go.Indicator(mode = "gauge+number+delta", value = musteri_verisi['Recency'], title = {'text': "Son Alışveriş (Gün)"}, delta = {'reference': segment_ortalamalari['Recency'], 'relative': False, 'decreasing': {'color': "green"}, 'increasing': {'color': "red"}}, gauge = {'axis': {'range': [0, segment_verisi['Recency'].max()]}, 'bar': {'color': "#ffbb28"}, 'steps' : [{'range': [0, segment_ortalamalari['Recency']], 'color': "lightgray"},], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': segment_ortalamalari['Recency']}}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50)); st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Kırmızı çizgi, '{musteri_segmenti}' segmentinin ortalama Recency'sini gösterir.")
    st.markdown("---")

    st.subheader("🗺️ Müşteri Yolculuğu Zaman Tüneli")
    if secilen_musteri in yolculuk_pivot.index:
        musteri_seyahati = yolculuk_pivot.loc[[secilen_musteri]].drop(columns=['CLV_Net_Kar'], errors='ignore').dropna(axis=1).T
        musteri_seyahati.columns = ['Segment']
        
        if musteri_seyahati.empty:
            st.info("Bu müşteri için henüz bir segment yolculuğu oluşmamış (tek bir dönemde aktif).")
        else:
            st.markdown(f"**{secilen_musteri}** adlı müşterinin zaman içindeki segment değişimi:")
            musteri_seyahati.index.name = "Dönem"
            # Hatalı satırı düzeltiyoruz:
            musteri_seyahati.index = musteri_seyahati.index.astype(str)
            st.dataframe(musteri_seyahati)
    else:
        st.info("Bu müşteri için yolculuk verisi bulunamadı.")

    st.markdown("---")
    st.subheader(f"📈 {secilen_musteri} için Satış Tahmini (Gelecek 6 Ay)")
    fig_tahmin = None
    musteri_ts = musteri_zaman_serisi_hazirla(temiz_df, secilen_musteri)
    if len(musteri_ts) >= 12:
        model, tahmin = satis_tahmini_yap(musteri_ts, ay_sayisi=6)
        fig_tahmin = tahmin_grafigini_ciz(model, tahmin, musteri_id=secilen_musteri, return_fig=True)
        st.pyplot(fig_tahmin)
    else:
        st.warning("Bu müşteri için yeterli geçmiş veri bulunmadığından satış tahmini yapılamadı.")

    st.markdown("---")
    st.subheader(f"🎁 {secilen_musteri} için Ürün Önerileri (Next Best Offer)")
    
    musteri_urunleri = temiz_df[temiz_df['MusteriID'] == secilen_musteri]['UrunKodu'].unique()
    tavsiyeler_df = urun_tavsiyesi_uret(birliktelik_kurallari, musteri_urunleri)
    
    if tavsiyeler_df.empty:
        st.info("Bu müşteri için şu anda otomatik bir ürün önerisi bulunmuyor.")
    else:
        st.markdown("Aşağıdaki ürünler, müşterinin geçmiş alımlarına dayanarak çapraz satış için tavsiye edilmektedir:")
        st.dataframe(tavsiyeler_df)
    
    st.markdown("---")
    st.subheader("📄 Raporlama")
    pdf_bytes = pdf_raporu_olustur(secilen_musteri, musteri_verisi, fig_tahmin, tavsiyeler_df)
    st.download_button(
        label="Bu Müşterinin Analiz Raporunu İndir (.pdf)",
        data=pdf_bytes,
        file_name=f"{secilen_musteri}_analiz_raporu.pdf",
        mime="application/pdf"
    )