# pages/12_Segmentasyon_Laboratuvari.py

import streamlit as st
import pandas as pd
import plotly.express as px

from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           kmeans_kumeleme_yap, hiyerarsik_kumeleme_yap, 
                           pca_ile_boyut_indirge, en_iyi_kume_sayisini_bul,
                           dinamik_kume_etiketle) # Eski fonksiyonu silip yenisini import ediyoruz

st.set_page_config(page_title="Segmentasyon Laboratuvarı", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    sonuclar_df = clv_df.copy()
    if 'MusteriAdi' not in sonuclar_df.columns:
        sonuclar_df['MusteriAdi'] = sonuclar_df.index
    return sonuclar_df

sonuclar_df = veriyi_getir_ve_isle()

st.title("🔬 Segmentasyon Laboratuvarı")
st.markdown("Farklı kümeleme algoritmalarını, parametreleri ve özellikleri deneyerek veri setiniz için en anlamlı segment yapısını bulun.")



with st.container(border=True):
    st.subheader("Laboratuvar Kontrolleri")
    
    kullanilabilir_ozellikler = ['Recency', 'Frequency', 'Monetary', 'CLV_Net_Kar', 'Churn_Olasiligi', 'MPS']
    secilen_ozellikler = st.multiselect(
        "Kümeleme için kullanılacak özellikleri seçin (en az 2):",
        options=kullanilabilir_ozellikler,
        default=['Recency', 'Frequency', 'Monetary']
    )
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        secilen_algoritma = st.selectbox("Kümeleme algoritmasını seçin:", ("K-Means", "Hiyerarşik Kümeleme"))
    with col_ctrl2:
        kume_sayisi = st.slider("Oluşturulacak Küme Sayısı:", 2, 10, 4, 1)
    
    if st.button(f"'{secilen_algoritma}' ile Analizi Çalıştır", type="primary", use_container_width=True):
        if len(secilen_ozellikler) < 2:
            st.error("Lütfen en az 2 özellik seçin.")
        else:
            st.session_state['run_analysis'] = True
            with st.spinner(f"{secilen_algoritma} algoritması çalıştırılıyor..."):
                sonuclar_pca_df = pca_ile_boyut_indirge(sonuclar_df.copy(), ozellikler=secilen_ozellikler)
                
                if secilen_algoritma == "K-Means":
                    kumeleme_sonucu_df, merkezler_df = kmeans_kumeleme_yap(sonuclar_pca_df, kume_sayisi, ozellikler=secilen_ozellikler)
                elif secilen_algoritma == "Hiyerarşik Kümeleme":
                    kumeleme_sonucu_df, merkezler_df = hiyerarsik_kumeleme_yap(sonuclar_pca_df, kume_sayisi, ozellikler=secilen_ozellikler)
            
            # --- GÜNCELLENMİŞ BÖLÜM: Her durumda dinamik etiketleme yapılıyor ---
            genel_ortalamalar = sonuclar_df[secilen_ozellikler].mean()
            kume_isimleri = dinamik_kume_etiketle(merkezler_df, genel_ortalamalar)

            kumeleme_sonucu_df['Kume_ID'] = kumeleme_sonucu_df['Kume']
            kumeleme_sonucu_df['Kume'] = kumeleme_sonucu_df['Kume_ID'].map(kume_isimleri)
            merkezler_df.index = merkezler_df.index.map(kume_isimleri)
            # --- GÜNCELLEME SONU ---

            st.session_state['kumeleme_sonucu_df'] = kumeleme_sonucu_df
            st.session_state['merkezler_df'] = merkezler_df
            st.session_state['secilen_ozellikler_cache'] = secilen_ozellikler
            st.session_state['kume_listesi'] = sorted(kumeleme_sonucu_df['Kume'].unique())
            st.session_state['secilen_kumeler'] = st.session_state['kume_listesi']

st.markdown("---")
st.header("✨ Optimal Segment Sayısı Bulucu")
st.markdown("Seçtiğiniz özelliklere göre en uygun segment sayısını 'Siluet Skoru' metriğini kullanarak otomatik olarak bulun.")

if st.button("Optimal Küme Sayısını Analiz Et"):
    if len(secilen_ozellikler) < 2:
        st.error("Lütfen en az 2 özellik seçin.")
    else:
        with st.spinner("Farklı küme sayıları için Siluet Skorları hesaplanıyor..."):
            skor_df = en_iyi_kume_sayisini_bul(sonuclar_df, ozellikler=secilen_ozellikler)
        st.session_state['skor_df'] = skor_df

if 'skor_df' in st.session_state:
    skor_df = st.session_state['skor_df']
    en_iyi_skor = skor_df.loc[skor_df['Siluet Skoru'].idxmax()]
    st.success(f"Analiz tamamlandı! En yüksek Siluet Skoru **{en_iyi_skor['Siluet Skoru']:.3f}** ile **{int(en_iyi_skor['Küme Sayısı'])}** kümede elde edildi.")
    fig_skor = px.line(skor_df, x="Küme Sayısı", y="Siluet Skoru", 
                       title="Farklı Küme Sayıları için Siluet Skorları", markers=True)
    fig_skor.add_vline(x=en_iyi_skor['Küme Sayısı'], line_dash="dash", line_color="red", annotation_text="En İyi Sonuç")
    st.plotly_chart(fig_skor, use_container_width=True)

if st.session_state.get('run_analysis', False):
    # ... (Bu bölümün içeriği bir önceki versiyonla aynı, sadece başlıklar ve etiketler güncellendi) ...
    st.markdown("---")
    st.header(f"Analiz Sonuçları")
    st.info(f"Kullanılan Özellikler: **{', '.join(st.session_state.secilen_ozellikler_cache)}**")
    sonuc_df = st.session_state['kumeleme_sonucu_df']
    merkezler = st.session_state['merkezler_df']
    st.subheader("Küme Profilleri (Ortalama Değerler)")
    st.dataframe(merkezler.style.format("{:,.1f}"))
    st.subheader("Küme Büyüklükleri")
    st.dataframe(sonuc_df['Kume'].value_counts().reset_index().rename(columns={'count':'Müşteri Sayısı'}))
    st.subheader("Kümelerin 2D Görselleştirmesi (PCA ile)")
    sonuc_df['Kume'] = sonuc_df['Kume'].astype('category')
    fig = px.scatter(sonuc_df, x='pca1', y='pca2', color='Kume',
                     hover_data=['MusteriAdi', 'Segment'],
                     title="Oluşturulan Müşteri Kümeleri",
                     labels={'pca1': 'Ana Bileşen 1', 'pca2': 'Ana Bileşen 2', 'Kume': 'Küme/Persona'})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Yeni Kümelerin Mevcut Segmentlerle İlişkisi")
    st.markdown("Bu tablo, oluşturduğunuz yeni kümelerin, standart RFM bazlı segmentlerle nasıl bir dağılım gösterdiğini ortaya koyar.")
    karsilastirma_tablosu = pd.crosstab(sonuc_df['Kume'], sonuc_df['Segment'])
    fig_heatmap = px.imshow(karsilastirma_tablosu, text_auto=True, aspect="auto",
                            labels=dict(x="Standart Segment", y="Yeni Küme/Persona", color="Müşteri Sayısı"),
                            title="Yeni Küme ve Standart Segment Kesişim Analizi")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    st.subheader("Detaylı Müşteri-Küme Listesi")
    
    secilen_kumeler = st.multiselect(
        "Görmek istediğiniz kümeleri/personaları seçin:",
        options=st.session_state.get('kume_listesi', []),
        key='secilen_kumeler'
    )

    if secilen_kumeler:
        gosterilecek_df = sonuc_df[sonuc_df['Kume'].isin(secilen_kumeler)]
        st.dataframe(gosterilecek_df[['MusteriAdi', 'Kume', 'Segment', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi']]
                     .rename(columns={'Kume': 'Yeni Küme/Persona'})
                     .style.format({
                        'MPS': '{:.0f}',
                        'CLV_Net_Kar': '{:,.0f} €',
                        'Churn_Olasiligi': '{:.1%}'
                     }))