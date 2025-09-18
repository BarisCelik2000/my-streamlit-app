# pages/6_Churn_Neden_Analizi.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           bireysel_churn_etkenlerini_hesapla)

st.set_page_config(page_title="Churn Neden Analizi", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri_guncellenmis.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, model, explainer, X, X_train, dogruluk = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    return clv_df, model, explainer, X, X_train
sonuclar_df, model, explainer, X, X_train = veriyi_getir_ve_isle()

st.title("🔍 Churn Neden Analizi (Random Forest + SHAP)")
st.markdown("""
Bu sayfa, müşterilerin neden churn ettiğini (kaybedildiğini) anlamak için daha gelişmiş bir makine öğrenmesi modelinin (`Random Forest`) içine **SHAP** kütüphanesi ile bakar. 
Bu yöntem, özelliklerin birbirleriyle olan karmaşık etkileşimlerini de dikkate alarak daha doğru ve güvenilir sonuçlar üretir.
""")



if model is None:
    st.warning("Churn neden analizi için yeterli veri bulunamadı veya model eğitilemedi.")
else:
    st.header("Genel Churn Etkenleri (SHAP Summary)")

    # --- YENİ EKLENEN BÖLÜM: Segment Filtresi ---
    st.markdown("---")
    segment_listesi = ['Tüm Müşteriler'] + sonuclar_df['Segment'].unique().tolist()
    secilen_segment = st.selectbox("Analiz edilecek müşteri segmentini seçin:", segment_listesi)
    
    # Seçilen segmente göre X_train verisini filtrele
    if secilen_segment == 'Tüm Müşteriler':
        X_train_filtrelenmis = X_train
    else:
        # Segmentteki müşterilerin ID'lerini (index) al
        segment_musteri_idler = sonuclar_df[sonuclar_df['Segment'] == secilen_segment].index
        # X_train'i bu ID'lere göre filtrele
        X_train_filtrelenmis = X_train[X_train.index.isin(segment_musteri_idler)]
    st.markdown("---")
    # --- YENİ BÖLÜM SONU ---
    
    if X_train_filtrelenmis.empty:
        st.warning(f"'{secilen_segment}' segmenti için eğitim verisinde yeterli örnek bulunamadı.")
    else:
        with st.spinner(f"'{secilen_segment}' segmenti için genel SHAP değerleri hesaplanıyor..."):
            if explainer:
                explanation = explainer(X_train_filtrelenmis)
                shap_values_for_churn = explanation[:,:,1]
            else:
                shap_values_for_churn = None

        if shap_values_for_churn is None:
            st.warning("SHAP değerleri hesaplanamadı.")
        else:
            st.markdown(f"Aşağıdaki grafik, **{secilen_segment}** için özelliklerin churn olasılığı üzerindeki genel etkisini özetler.")
            
            fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
            shap.plots.bar(shap_values_for_churn, show=False)
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("📊 Grafiği Nasıl Yorumlanır?"):
                st.info("""
                **SHAP Değeri:** Bir özelliğin, bir tahmin üzerindeki ortalama mutlak etkisini gösterir. Çubuk ne kadar uzunsa, o özelliğin churn tahminindeki rolü o kadar büyüktür.
                """)
        
    st.markdown("---")
    
    st.header("Bireysel Müşteri Risk Analizi")
    
    riskli_musteriler_listesi = sonuclar_df[sonuclar_df['Churn_Olasiligi'] > 0.5].index
    if len(riskli_musteriler_listesi) == 0:
        st.info("Analiz edilecek yüksek riskli müşteri bulunmuyor.")
    else:
        secilen_musteri = st.selectbox("Analiz edilecek yüksek riskli bir müşteri seçin:", riskli_musteriler_listesi)

        if secilen_musteri:
            musteri_verisi_X = X.loc[[secilen_musteri]]
            
            with st.spinner(f"'{secilen_musteri}' için SHAP dökümü hesaplanıyor..."):
                explanation_bireysel = bireysel_churn_etkenlerini_hesapla(explainer, musteri_verisi_X)

            st.subheader(f"'{secilen_musteri}' için Risk Dökümü (SHAP Waterfall)")
            
            if explanation_bireysel is not None:
                fig_waterfall, ax_waterfall = plt.subplots()
                shap.plots.waterfall(explanation_bireysel[0, :, 1], show=False)
                plt.tight_layout()
                st.pyplot(fig_waterfall)
            
                with st.expander("📊 Şelale Grafiği Nasıl Yorumlanır?"):
                    st.info("""
                    Bu grafik, seçilen müşterinin churn olasılığının nasıl oluştuğunu adım adım gösterir.
                    - **E[f(x)] (En Alttaki Gri Çubuk):** Modelin ortalama (beklenen) başlangıç tahminidir.
                    - **Kırmızı Çubuklar:** Churn olasılığını **artıran** faktörlerdir.
                    - **Mavi Çubuklar:** Churn olasılığını **azaltan** faktörlerdir.
                    - **f(x) (En Üstteki Gri Çubuk):** Tüm faktörlerin etkisi toplandıktan sonra ulaşılan nihai tahmin skorudur.
                    """)
                st.markdown("---")
                st.subheader("💡 Aksiyon Önerisi")

                # Müşterinin churn skorunu en çok artıran faktörü bul
                shap_values = explanation_bireysel[0, :, 1].values
                feature_names = explanation_bireysel.feature_names
                
                # Sadece pozitif (churn'ü artıran) SHAP değerlerini dikkate al
                pozitif_etkiler = {name: val for name, val in zip(feature_names, shap_values) if val > 0}
                
                if not pozitif_etkiler:
                    st.success("Bu müşteri için önemli bir risk faktörü tespit edilmedi.")
                else:
                    # En büyük risk faktörünü bul
                    en_buyuk_risk_faktoru = max(pozitif_etkiler, key=pozitif_etkiler.get)
                    
                    if en_buyuk_risk_faktoru == 'Recency':
                        st.warning(f"**En Büyük Risk Faktörü: Recency (Son Alışveriş Tarihi)**")
                        st.info(f"**Öneri:** Bu müşteri uzun süredir alışveriş yapmıyor. Onu tekrar kazanmak için kişiselleştirilmiş bir **'Sizi Özledik!'** e-postası veya SMS ile cazip bir indirim sunmayı düşünebilirsiniz.")
                    elif en_buyuk_risk_faktoru == 'Frequency':
                        st.warning(f"**En Büyük Risk Faktörü: Frequency (Alışveriş Sıklığı)**")
                        st.info(f"**Öneri:** Müşterinin alışveriş sıklığı beklentinin altında. Sadakat programı puanları, çoklu alım indirimleri veya abonelik modelleri ile **tekrar eden alışverişleri teşvik etmeyi** deneyebilirsiniz.")
                    elif en_buyuk_risk_faktoru == 'Monetary':
                        st.warning(f"**En Büyük Risk Faktörü: Monetary (Harcama Tutarı)**")
                        st.info(f"**Öneri:** Müşterinin ortalama harcama tutarı düşük. Değerini artırmak için **üst satış (up-sell)** veya **tamamlayıcı ürünlerle çapraz satış (cross-sell)** fırsatları sunabilirsiniz.")
