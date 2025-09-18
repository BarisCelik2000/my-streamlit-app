# pages/11_Musteri_Benzerlik_Analizi.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, 
                           musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, 
                           clv_hesapla,
                           musteri_benzerlik_hesapla,
                           benzer_musteri_urun_onerileri,
                           segmente_benzer_musteri_bul,
                           urun_benzerligi_hesapla)

st.set_page_config(page_title="Müşteri Benzerlik Analizi", layout="wide")

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

st.title("👥 Müşteri Benzerlik Analizi (Look-alike)")
st.markdown("""
Bu araç, seçtiğiniz bir kaynak müşteriye veya segmente en çok benzeyen diğer müşterileri bularak yeni hedef kitleler oluşturmanıza yardımcı olur.
""")



st.markdown("---")
analiz_tipi = st.radio(
    "Hangi tipte bir benzerlik analizi yapmak istersiniz?",
    ("Tek Bir Müşteriye Göre", "Bir Segmente Göre"),
    horizontal=True
)
st.markdown("---")

if analiz_tipi == "Tek Bir Müşteriye Göre":
    @st.cache_data
    def benzerlik_matrisini_getir(_sonuclar_df):
        return musteri_benzerlik_hesapla(_sonuclar_df)

    similarity_df = benzerlik_matrisini_getir(sonuclar_df)

    st.header("Benzer Müşterileri Bul")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        secilen_musteri = st.selectbox("Bir kaynak müşteri seçin:", sonuclar_df.index)
    with col2:
        top_n = st.slider("Kaç adet benzer müşteri bulunsun?", 1, 10, 5, key="topn_tekil")

    st.subheader("Gelişmiş Benzerlik Ayarları")
    use_product_similarity = st.toggle("Ürün Zevki Benzerliğini Aktif Et", value=False)
    
    rfm_agirlik = 1.0
    if use_product_similarity:
        rfm_agirlik = st.slider("Benzerlik Ağırlığı (RFM vs. Ürün Zevki)", 0.0, 1.0, 0.7, 0.05,
                                help="1.0'a yaklaştıkça RFM (davranış) benzerliği, 0.0'a yaklaştıkça Ürün Zevki benzerliği önceliklendirilir.")
    urun_agirlik = 1.0 - rfm_agirlik
    
    if secilen_musteri:
        rfm_benzerlik_skorlari = similarity_df[secilen_musteri]
        
        if use_product_similarity:
            with st.spinner("Ürün zevki benzerlikleri hesaplanıyor..."):
                urun_benzerlik_skorlari = urun_benzerligi_hesapla(temiz_df, secilen_musteri)
            
            birlesik_skor_df = pd.DataFrame({'RFM_Skor': rfm_benzerlik_skorlari, 'Urun_Skor': urun_benzerlik_skorlari}).dropna()
            
            if not birlesik_skor_df.empty:
                scaler = MinMaxScaler()
                birlesik_skor_df[['RFM_Skor', 'Urun_Skor']] = scaler.fit_transform(birlesik_skor_df[['RFM_Skor', 'Urun_Skor']])
                birlesik_skor_df['Nihai_Skor'] = (rfm_agirlik * birlesik_skor_df['RFM_Skor']) + (urun_agirlik * birlesik_skor_df['Urun_Skor'])
                nihai_skorlar = birlesik_skor_df['Nihai_Skor']
                gosterilecek_tablo = birlesik_skor_df.rename(columns={'Nihai_Skor': 'Benzerlik_Skoru'})
            else:
                nihai_skorlar = pd.Series(dtype='float64') # Boş seri oluştur
                gosterilecek_tablo = pd.DataFrame()
        else:
            nihai_skorlar = rfm_benzerlik_skorlari
            gosterilecek_tablo = pd.DataFrame({'Benzerlik_Skoru': rfm_benzerlik_skorlari})
        
        look_alike_musteriler = nihai_skorlar.drop(secilen_musteri, errors='ignore').nlargest(top_n)
        
        st.markdown("---")
        st.subheader(f"'{secilen_musteri}' Adlı Müşteriye En Çok Benzeyenler:")
        
        if look_alike_musteriler.empty:
            st.warning("Seçilen ayarlarla benzer müşteri bulunamadı.")
        else:
            benzer_musteri_detaylari = sonuclar_df.loc[look_alike_musteriler.index].join(gosterilecek_tablo)
            kaynak_musteri_detaylari = sonuclar_df.loc[[secilen_musteri]].copy()
            kaynak_musteri_detaylari.reset_index(inplace=True)

            if use_product_similarity:
                st.dataframe(benzer_musteri_detaylari[['Segment', 'Benzerlik_Skoru', 'RFM_Skor', 'Urun_Skor']]
                             .sort_values('Benzerlik_Skoru', ascending=False)
                             .style.format({
                                 'Benzerlik_Skoru': '{:.2%}',
                                 'RFM_Skor': '{:.2%}',
                                 'Urun_Skor': '{:.2%}'
                             }).background_gradient(cmap='Greens', subset=['Benzerlik_Skoru']))
            else:
                st.dataframe(benzer_musteri_detaylari[['Segment', 'Benzerlik_Skoru']]
                             .sort_values('Benzerlik_Skoru', ascending=False)
                             .style.format({'Benzerlik_Skoru': '{:.2%}'}).background_gradient(cmap='Greens', subset=['Benzerlik_Skoru']))

            st.markdown("**RFM Profili Karşılaştırması (Skor Bazlı)**")
            karsilastirma_df = pd.concat([kaynak_musteri_detaylari, benzer_musteri_detaylari.reset_index()], ignore_index=True)
            fig = go.Figure()
            categories = ['Recency Skoru', 'Frequency Skoru', 'Monetary Skoru']
            for i, row in karsilastirma_df.iterrows():
                musteri_adi = row['MusteriID']
                fig.add_trace(go.Scatterpolar(r=[row['R_Score'], row['F_Score'], row['M_Score']], theta=categories, fill='toself' if musteri_adi == secilen_musteri else 'none', name=musteri_adi, line=dict(width=4 if musteri_adi == secilen_musteri else 2)))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader(f"'{secilen_musteri}' İçin Ürün Önerileri")
            with st.spinner("Benzer müşterilerin sepetleri analiz ediliyor..."):
                urun_onerileri_df = benzer_musteri_urun_onerileri(temiz_df, secilen_musteri, look_alike_musteriler.index.tolist())
            if urun_onerileri_df.empty:
                st.info("Bu müşteri profiline uygun yeni bir ürün önerisi bulunamadı.")
            else:
                st.dataframe(urun_onerileri_df.style.format({'Alim_Orani': '{:.1f}%'}).background_gradient(cmap='Blues', subset=['Alim_Orani']))


elif analiz_tipi == "Bir Segmente Göre":
    st.header("Bir Segmente Benzeyen Hedef Kitle Oluştur")
    st.markdown("Bir kaynak segment seçin. Araç, bu segmentin ortalama profiline en çok benzeyen ve **henüz o segmentte olmayan** müşterileri bulacaktır.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        segment_listesi = sonuclar_df['Segment'].unique().tolist()
        if not segment_listesi:
            st.warning("Analiz için segment bulunamadı.")
        else:
            kaynak_segment = st.selectbox("Kaynak segmenti seçin:", segment_listesi)
    with col2:
        top_n_segment = st.slider("Kaç adet benzer müşteri bulunsun?", 10, 200, 50, key="topn_segment")

    if 'kaynak_segment' in locals() and st.button("Benzer Kitleyi Bul", type="primary"):
        with st.spinner(f"'{kaynak_segment}' segmentine benzeyen müşteriler analiz ediliyor..."):
            benzer_kitle_df = segmente_benzer_musteri_bul(sonuclar_df, kaynak_segment, top_n=top_n_segment)
        
        st.session_state.benzer_kitle_df = benzer_kitle_df
        st.session_state.kaynak_segment = kaynak_segment
    
    if 'benzer_kitle_df' in st.session_state:
        benzer_kitle_df = st.session_state.benzer_kitle_df
        kaynak_segment = st.session_state.kaynak_segment

        st.markdown("---")
        st.subheader(f"'{kaynak_segment}' Segmentine En Çok Benzeyen Potansiyel Müşteriler")
        
        if benzer_kitle_df.empty:
            st.warning("Bu segmente benzeyen başka müşteri bulunamadı.")
        else:
            st.info(f"Aşağıdaki liste, **'{kaynak_segment}'** segmentine en çok benzeyen ancak şu anda farklı segmentlerde olan müşterileri göstermektedir. Bu kitle, bir üst segmente taşımak için ideal bir hedeftir.")
            st.dataframe(benzer_kitle_df[['Segment', 'Recency', 'Frequency', 'Monetary', 'Benzerlik_Skoru']]
                         .sort_values('Benzerlik_Skoru', ascending=False)
                         .style.format({'Benzerlik_Skoru': '{:.2%}', 'Monetary': '{:,.0f} €'})
                         .background_gradient(cmap='Greens', subset=['Benzerlik_Skoru']))