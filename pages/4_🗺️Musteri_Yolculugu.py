# pages/4_Müşteri_Yolculuğu.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import musteri_yolculugu_analizi_yap, rfm_skorlarini_hesapla, musterileri_segmentle, clv_hesapla, churn_tahmin_modeli_olustur

st.set_page_config(page_title="Müşteri Yolculuğu", layout="wide")

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

st.title("🗺️ Müşteri Yaşam Döngüsü Analizi")
st.markdown("""
Bu sayfa, müşterilerinizin **kazanım, segmentler arası geçiş ve kayıp (churn)** süreçlerini içeren tam yaşam döngüsünü görselleştirir.
""")

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

with st.spinner('Müşteri yolculukları hesaplanıyor... Bu işlem birkaç dakika sürebilir.'):
    yolculuk_pivot, _, _, _ = musteri_yolculugu_analizi_yap(temiz_df, sonuclar_df)

st.success("Analiz tamamlandı!")

st.markdown("---")

st.header("Analiz Metriğini Seçin")
analiz_tipi = st.radio(
    "Akış diyagramı neyi temsil etsin?",
    ("Müşteri Sayısı", "Toplam Yaşam Boyu Değeri (CLV)"),
    horizontal=True
)
st.header("Karşılaştırma Dönemlerini Seçin")

if yolculuk_pivot.empty or len([c for c in yolculuk_pivot.columns if c != 'CLV_Net_Kar']) < 2:
    st.warning("Segment geçişlerini karşılaştırmak için yeterli zaman periyodu (en az 2 çeyrek) bulunamadı.")
else:
    donemler = sorted([c for c in yolculuk_pivot.columns if c != 'CLV_Net_Kar'], reverse=True)
    
    col1, col2 = st.columns(2)
    with col1:
        secilen_son_donem = st.selectbox("Bitiş Dönemi", donemler, index=0)
    with col2:
        secilen_onceki_donem = st.selectbox("Başlangıç Dönemi", donemler, index=1)

    if secilen_onceki_donem and secilen_son_donem and secilen_onceki_donem != secilen_son_donem:
        
        df_donemler = yolculuk_pivot[[secilen_onceki_donem, secilen_son_donem, 'CLV_Net_Kar']].copy()
        
        mevcut_musteriler = df_donemler.dropna(subset=[secilen_onceki_donem, secilen_son_donem]).copy()
        mevcut_musteriler.rename(columns={secilen_onceki_donem: 'Onceki_Segment', secilen_son_donem: 'Simdiki_Segment'}, inplace=True)

        yeni_musteriler = df_donemler[df_donemler[secilen_onceki_donem].isna() & df_donemler[secilen_son_donem].notna()].copy()
        yeni_musteriler['Onceki_Segment'] = 'Yeni Müşteri'
        yeni_musteriler.rename(columns={secilen_son_donem: 'Simdiki_Segment'}, inplace=True)

        kayip_musteriler = df_donemler[df_donemler[secilen_onceki_donem].notna() & df_donemler[secilen_son_donem].isna()].copy()
        kayip_musteriler['Simdiki_Segment'] = 'Pasif / Churn'
        kayip_musteriler.rename(columns={secilen_onceki_donem: 'Onceki_Segment'}, inplace=True)
        
        gecis_df_tam = pd.concat([
            mevcut_musteriler[['Onceki_Segment', 'Simdiki_Segment', 'CLV_Net_Kar']],
            yeni_musteriler[['Onceki_Segment', 'Simdiki_Segment', 'CLV_Net_Kar']],
            kayip_musteriler[['Onceki_Segment', 'Simdiki_Segment', 'CLV_Net_Kar']]
        ])
        
        onceki_donem_str = str(secilen_onceki_donem)
        son_donem_str = str(secilen_son_donem)
        
        st.header(f"Yaşam Döngüsü Akışı ({onceki_donem_str} -> {son_donem_str})")
        
        if analiz_tipi == "Müşteri Sayısı":
            gecis_df_sankey = gecis_df_tam.groupby(['Onceki_Segment', 'Simdiki_Segment']).size().reset_index(name='deger')
            deger_formati_metin = "müşteri"
            deger_formati_gorsel = ".0f"
            title_text = f"{onceki_donem_str} ile {son_donem_str} Arası Müşteri Sayısı Akışı"
        else:
            gecis_df_sankey = gecis_df_tam.groupby(['Onceki_Segment', 'Simdiki_Segment'])['CLV_Net_Kar'].sum().reset_index(name='deger')
            deger_formati_metin = "€ CLV"
            deger_formati_gorsel = ",.0f €"
            title_text = f"{onceki_donem_str} ile {son_donem_str} Arası Toplam CLV Akışı (€)"

        segment_siralama = ['Yeni Müşteri', 'Şampiyonlar', 'Potansiyel Şampiyonlar', 'Sadık Müşteriler', 'Riskli Müşteriler', 'Kayıp Müşteriler', 'Pasif / Churn']
        segment_renkleri = {
            'Yeni Müşteri': '#4CAF50', 'Pasif / Churn': '#F44336',
            'Kayıp Müşteriler': '#E57373', 'Riskli Müşteriler': '#FFB74D', 'Sadık Müşteriler': '#9CCC65',
            'Potansiyel Şampiyonlar': '#64B5F6', 'Şampiyonlar': '#BA68C8'
        }
        
        tum_etiketler = sorted(pd.concat([gecis_df_sankey['Onceki_Segment'], gecis_df_sankey['Simdiki_Segment']]).unique(), 
                                 key=lambda x: segment_siralama.index(x) if x in segment_siralama else len(segment_siralama))
        
        etiket_map = {etiket: i for i, etiket in enumerate(tum_etiketler)}
        node_colors = [segment_renkleri.get(etiket, '#CCCCCC') for etiket in tum_etiketler]
        link_colors = [segment_renkleri.get(row['Onceki_Segment'], '#A0A0A0') for _, row in gecis_df_sankey.iterrows()]
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=tum_etiketler, color=node_colors),
            link=dict(
                source=gecis_df_sankey['Onceki_Segment'].map(etiket_map), 
                target=gecis_df_sankey['Simdiki_Segment'].map(etiket_map), 
                value=gecis_df_sankey['deger'],
                color=link_colors,
                hovertemplate='%{source.label} -> %{target.label}<br>Değer: %{value:' + deger_formati_gorsel + '}<extra></extra>'
            )
        )])
        fig_sankey.update_layout(title_text=title_text, font_size=12, height=600)
        st.plotly_chart(fig_sankey, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Otomatik Analiz Özeti")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            yeni_musteri_akisi = gecis_df_sankey[gecis_df_sankey['Onceki_Segment'] == 'Yeni Müşteri']
            if not yeni_musteri_akisi.empty:
                en_buyuk_kazanim = yeni_musteri_akisi.loc[yeni_musteri_akisi['deger'].idxmax()]
                st.success(f"**Yeni Kazanım:** En çok yeni müşteri **{en_buyuk_kazanim['Simdiki_Segment']}** segmentine dahil oldu ({en_buyuk_kazanim['deger']:,.0f} {deger_formati_metin}).")

            segment_degerleri = {'Şampiyonlar': 5, 'Potansiyel Şampiyonlar': 4, 'Sadık Müşteriler': 3, 'Riskli Müşteriler': 2, 'Kayıp Müşteriler': 1}
            ic_gecisler = gecis_df_sankey[~gecis_df_sankey['Onceki_Segment'].isin(['Yeni Müşteri']) & ~gecis_df_sankey['Simdiki_Segment'].isin(['Pasif / Churn'])].copy()
            ic_gecisler['onceki_deger'] = ic_gecisler['Onceki_Segment'].map(segment_degerleri).astype(float)
            ic_gecisler['simdiki_deger'] = ic_gecisler['Simdiki_Segment'].map(segment_degerleri).astype(float)
            pozitif_akislar = ic_gecisler[ic_gecisler['simdiki_deger'] > ic_gecisler['onceki_deger']]
            if not pozitif_akislar.empty:
                en_iyi_gecis = pozitif_akislar.loc[pozitif_akislar['deger'].idxmax()]
                st.info(f"**En İyi Gelişme:** En büyük pozitif segment geçişi **{en_iyi_gecis['Onceki_Segment']}** → **{en_iyi_gecis['Simdiki_Segment']}** arasında yaşandı ({en_iyi_gecis['deger']:,.0f} {deger_formati_metin}).")

        with col_insight2:
            kayip_musteri_akisi = gecis_df_sankey[gecis_df_sankey['Simdiki_Segment'] == 'Pasif / Churn']
            if not kayip_musteri_akisi.empty:
                en_buyuk_kayip = kayip_musteri_akisi.loc[kayip_musteri_akisi['deger'].idxmax()]
                st.error(f"**En Kritik Kayıp:** En çok müşteri **{en_buyuk_kayip['Onceki_Segment']}** segmentinden kaybedildi ({en_buyuk_kayip['deger']:,.0f} {deger_formati_metin}).")
            
            negatif_akislar = ic_gecisler[ic_gecisler['simdiki_deger'] < ic_gecisler['onceki_deger']]
            if not negatif_akislar.empty:
                en_kotu_gecis = negatif_akislar.loc[negatif_akislar['deger'].idxmax()]
                st.warning(f"**Dikkat:** En büyük negatif segment geçişi **{en_kotu_gecis['Onceki_Segment']}** → **{en_kotu_gecis['Simdiki_Segment']}** arasında yaşandı ({en_kotu_gecis['deger']:,.0f} {deger_formati_metin}).")
            
    else:
        st.error("Lütfen karşılaştırma için birbirinden farklı iki dönem seçin.")

st.markdown("---")
st.header("Bireysel Müşteri Yolculuğu")
bireysel_yolculuk_pivot = yolculuk_pivot.drop(columns=['CLV_Net_Kar'], errors='ignore')
secilen_musteri_yolculuk = st.selectbox("Yolculuğunu görmek için bir müşteri seçin:", bireysel_yolculuk_pivot.index)

if secilen_musteri_yolculuk:
    musteri_seyahati = bireysel_yolculuk_pivot.loc[[secilen_musteri_yolculuk]].dropna(axis=1).T
    musteri_seyahati.columns = ['Segment']
    
    if musteri_seyahati.empty:
        st.warning("Seçilen müşteri için yolculuk verisi bulunamadı.")
    else:
        st.markdown(f"**{secilen_musteri_yolculuk}** adlı müşterinin zaman içindeki segment değişimi:")
        musteri_seyahati.index.name = "Dönem"
        musteri_seyahati.index = musteri_seyahati.index.astype(str)
        st.dataframe(musteri_seyahati)