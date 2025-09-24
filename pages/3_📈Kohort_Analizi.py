# pages/3_Kohort_Analizi.py

import streamlit as st
import plotly.express as px
import pandas as pd 
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import kohort_analizi_yap

st.set_page_config(page_title="Kohort Analizi", layout="wide")

@st.cache_data
def veriyi_getir():
    dosya_adi = 'satis_verileri_guncellenmis.json'
    return veriyi_yukle_ve_temizle(dosya_adi)

st.title("📈 Kohort Analizi")
st.markdown("Bu analiz, müşterilerinizi başlangıç tarihlerine (kohortlarına) göre gruplar ve zaman içindeki davranışlarını farklı metriklere göre gösterir.")

temiz_df = veriyi_getir()

st.markdown("---")
st.subheader("Analiz Dönemini Seçin")
min_tarih = temiz_df['Tarih'].min().date()
max_tarih = temiz_df['Tarih'].max().date()

col_tarih1, col_tarih2 = st.columns(2)
with col_tarih1:
    baslangic_tarihi = st.date_input("Başlangıç Tarihi", min_tarih, min_value=min_tarih, max_value=max_tarih, key="kohort_start")
with col_tarih2:
    bitis_tarihi = st.date_input("Bitiş Tarihi", max_tarih, min_value=min_tarih, max_value=max_tarih, key="kohort_end")

# Ana DataFrame'i seçilen tarihlere göre filtrele
filtrelenmis_df = temiz_df[
    (temiz_df['Tarih'].dt.date >= baslangic_tarihi) & 
    (temiz_df['Tarih'].dt.date <= bitis_tarihi)
]

st.markdown("---")
st.subheader("Analiz Parametreleri")
col1, col2 = st.columns(2)

with col1:
    metrik_secenekleri = {
        "Elde Tutma Oranı (%)": "retention",
        "Müşteri Başına Ortalama Harcama (€)": "avg_spend"
    }
    secilen_metrik_adi = st.selectbox("Görüntülemek İstediğiniz Metriği Seçin:", metrik_secenekleri.keys())
    secilen_metrik_kodu = metrik_secenekleri[secilen_metrik_adi]

with col2:
    periyot_secenekleri = {
        "Aylık": "M",
        "Çeyreklik": "Q"
    }
    secilen_periyot_adi = st.radio("Kohort Zaman Aralığını Seçin:", periyot_secenekleri.keys(), horizontal=True)
    secilen_periyot_kodu = periyot_secenekleri[secilen_periyot_adi]
st.markdown("---")

with st.spinner(f"'{secilen_periyot_adi}' periyotlar ve '{secilen_metrik_adi}' metriği için kohort analizi yapılıyor..."):
    heatmap_matrix = kohort_analizi_yap(filtrelenmis_df, metric=secilen_metrik_kodu, period=secilen_periyot_kodu)


st.success("Analiz tamamlandı!")

if not heatmap_matrix.empty:
    donem_turu = "Ay" if secilen_periyot_kodu == 'M' else "Çeyrek"
    
    if secilen_metrik_kodu == 'retention':
        format_str = ".0%"
        color_label = "Elde Tutma Oranı"
    else:
        format_str = ",.0f"
        color_label = "Ortalama Harcama (€)"

    fig = px.imshow(heatmap_matrix, text_auto=format_str, aspect="auto",
                    labels=dict(x=f"Aktiviteden Sonra Geçen {donem_turu}", y="Müşteri Kohortu", color=color_label),
                    title=f"Kohort Bazında {secilen_metrik_adi}")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 Tablodaki her bir satır, o {donem_turu.lower()} başlayan müşteri grubunu temsil eder. Sütunlar ise o grubun seçtiğiniz metrikteki performansını gösterir.")

    if secilen_metrik_kodu == 'retention':
        st.markdown("---")
        st.subheader("Kohortların Karşılaştırmalı Performansı")
        
        col1, col2 = st.columns(2)

        if 6 in heatmap_matrix.columns:
            with col1:
                ortalama_6ay_elde_tutma = heatmap_matrix[6].mean()
                st.metric(
                    label="Ortalama 6. Ay Elde Tutma Oranı",
                    value=f"{ortalama_6ay_elde_tutma:.1%}"
                )
            with col2:
                en_iyi_kohort_6ay = heatmap_matrix[6].idxmax()
                en_iyi_deger_6ay = heatmap_matrix[6].max()
                st.metric(
                    label="En İyi Performanslı Kohort (6. Ay)",
                    value=en_iyi_kohort_6ay,
                    help=f"Bu kohort, 6. ayda %{en_iyi_deger_6ay:.1f} elde tutma oranına sahipti."
                )
        else:
            st.info("6 aylık veriye sahip yeterli kohort bulunmadığı için özet KPI'lar gösterilemiyor.")

        df_line = heatmap_matrix.copy()
        df_line.index.name = 'Kohort'
        df_line = df_line.reset_index()
        df_line_melted = df_line.melt(id_vars='Kohort', var_name='Ay_Indeksi', value_name='EldeTutmaOrani')
        df_line_melted.dropna(subset=['EldeTutmaOrani'], inplace=True)

        fig_line = px.line(
            df_line_melted,
            x='Ay_Indeksi',
            y='EldeTutmaOrani',
            color='Kohort',
            title='Tüm Kohortların Elde Tutma Oranı Eğrileri',
            labels={'Ay_Indeksi': 'Kazanıldıktan Sonra Geçen Ay', 'EldeTutmaOrani': 'Elde Tutma Oranı (%)'},
            markers=True
        )
        fig_line.update_layout(yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("Bu grafik, hangi müşteri grubunun (kohortun) zamanla daha sadık kaldığını doğrudan karşılaştırmanızı sağlar.")
    
else:
    st.warning("Kohort analizi için yeterli veri bulunamadı.")