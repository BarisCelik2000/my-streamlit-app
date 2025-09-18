# pages/1_Ürün_Analizi.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import networkx as nx # Yeni eklenen kütüphane

# Gerekli fonksiyonları merkezi modüllerden import edelim
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           market_basket_analizi_yap, 
                           urun_performans_analizi_yap,
                           urun_icin_segment_profili, segment_icin_urun_profili, sayfa_raporu_olustur)

st.set_page_config(page_title="Ürün Analizi", layout="wide")

# Her sayfanın kendi veri getirme fonksiyonunu tanımlaması en sağlam yöntemdir.
@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.json'
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    
    return temiz_df, clv_df

# --- ANA SAYFA ---
st.title("📈 Ürün Analizi")
st.markdown("Bu modül, ürünlerinizin bireysel performanslarını, sepet birlikteliklerini ve müşteri segmentleriyle olan ilişkilerini analiz eder.")

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

tab1, tab2, tab3 = st.tabs(["Ürün Performans Panosu", "Pazar Sepeti Analizi (Birliktelik)", "Ürün-Segment Profili"])

# --- SEKME 1: ÜRÜN PERFORMANS PANOSU ---
with tab1:
    st.header("Ürün Performans Panosu")
    st.markdown("Ürünlerinizin ciro, karlılık ve popülerlik bazında performansını inceleyin.")

    
    st.markdown("---")
    st.subheader("Analiz Dönemini Seçin")
    
    min_tarih = temiz_df['Tarih'].min().date()
    max_tarih = temiz_df['Tarih'].max().date()
    
    col_tarih1, col_tarih2 = st.columns(2)
    with col_tarih1:
        secilen_baslangic_tarihi = st.date_input("Başlangıç Tarihi", min_tarih, min_value=min_tarih, max_value=max_tarih)
    with col_tarih2:
        secilen_bitis_tarihi = st.date_input("Bitiş Tarihi", max_tarih, min_value=min_tarih, max_value=max_tarih)
    st.markdown("---")

    with st.spinner(f"{secilen_baslangic_tarihi} ve {secilen_bitis_tarihi} arası için ürün performansları hesaplanıyor..."):
        performans_df = urun_performans_analizi_yap(temiz_df, secilen_baslangic_tarihi, secilen_bitis_tarihi)
    
    if performans_df.empty:
        st.warning("Seçilen dönem için performans analizi yapılacak yeterli ürün verisi bulunamadı.")
    else:
        st.subheader(f"Seçilen Dönemin En İyi Performans Gösteren Ürünleri")
        col1, col2, col3 = st.columns(3)
        en_iyi_ciro = performans_df.loc[performans_df['Toplam_Ciro'].idxmax()]
        en_iyi_kar = performans_df.loc[performans_df['Toplam_Net_Kar'].idxmax()]
        en_populer = performans_df.loc[performans_df['Benzersiz_Musteri_Sayisi'].idxmax()]
        
        col1.metric("En Yüksek Cirolu Ürün", en_iyi_ciro['UrunKodu'][:30]+"...", f"{en_iyi_ciro['Toplam_Ciro']:,.0f} €")
        col2.metric("En Yüksek Karlı Ürün", en_iyi_kar['UrunKodu'][:30]+"...", f"{en_iyi_kar['Toplam_Net_Kar']:,.0f} €")
        col3.metric("En Popüler Ürün (Müşteri Sayısı)", en_populer['UrunKodu'][:30]+"...", f"{en_populer['Benzersiz_Musteri_Sayisi']:.0f} Müşteri")
        
        st.markdown("---")
        
        st.subheader("Tüm Ürünlerin Performans Detayları (ABC Analizi Dahil)")
        
        df_abc = performans_df.sort_values(by='Toplam_Ciro', ascending=False).reset_index(drop=True)
        df_abc['Kümülatif_Ciro'] = df_abc['Toplam_Ciro'].cumsum()
        df_abc['Kümülatif_%'] = 100 * df_abc['Kümülatif_Ciro'] / df_abc['Toplam_Ciro'].sum()

        def abc_class(x):
            if x <= 80: return "A (%80 ciro)"
            elif x <= 95: return "B (sonraki %15)"
            else: return "C (kalan %5)"

        df_abc['ABC_Sınıfı'] = df_abc['Kümülatif_%'].apply(abc_class)
        
        sec_abc = st.multiselect(
            "Gösterilecek ABC sınıfları seçin:",
            options=["A (%80 ciro)", "B (sonraki %15)", "C (kalan %5)"],
            default=["A (%80 ciro)", "B (sonraki %15)", "C (kalan %5)"]
        )

        if sec_abc: df_abc_filtreli = df_abc[df_abc['ABC_Sınıfı'].isin(sec_abc)]
        else: df_abc_filtreli = df_abc

        st.dataframe(df_abc_filtreli.style.format({
            'Toplam_Ciro': '{:,.0f} €', 'Toplam_Net_Kar': '{:,.0f} €', 'Kar_Marji': '{:.1f}%',
            'Kümülatif_Ciro': '{:,.0f} €', 'Kümülatif_%': '{:.1f}%'
        }))

        st.markdown("---")
        st.subheader("Ürün Portföy Analizi (Stratejik Gruplama)")
        st.markdown("Ürünlerinizi satış adedi ve kar marjına göre 4 stratejik gruba ayırarak portföyünüzü görselleştirin.")

        df_portfolio = df_abc_filtreli.dropna(subset=['Toplam_Satis_Adedi', 'Kar_Marji'])
        df_portfolio = df_portfolio[(df_portfolio['Toplam_Satis_Adedi'] > 0) & (df_portfolio['Kar_Marji'] > 0)]

        if not df_portfolio.empty:
            ortalama_satis_adedi = df_portfolio['Toplam_Satis_Adedi'].median()
            ortalama_kar_marji = df_portfolio['Kar_Marji'].median()

            def portfoy_grubu_ata(row):
                if row['Toplam_Satis_Adedi'] >= ortalama_satis_adedi and row['Kar_Marji'] >= ortalama_kar_marji:
                    return 'Yıldızlar'
                elif row['Toplam_Satis_Adedi'] < ortalama_satis_adedi and row['Kar_Marji'] >= ortalama_kar_marji:
                    return 'Soru İşaretleri'
                elif row['Toplam_Satis_Adedi'] >= ortalama_satis_adedi and row['Kar_Marji'] < ortalama_kar_marji:
                    return 'Nakit İnekleri'
                else:
                    return 'Zayıflar'

            df_portfolio['Portföy_Grubu'] = df_portfolio.apply(portfoy_grubu_ata, axis=1)

            fig_portfolio = px.scatter(
                df_portfolio, x="Toplam_Satis_Adedi", y="Kar_Marji", size="Toplam_Ciro", color="Portföy_Grubu",
                hover_name="UrunKodu", log_x=True, size_max=60, title="Ürün Portföyü Dağılım Grafiği",
                labels={'Toplam_Satis_Adedi': 'Toplam Satış Adedi (Log Ölçek)', 'Kar_Marji': 'Kar Marjı (%)'}
            )
            fig_portfolio.add_vline(x=ortalama_satis_adedi, line_dash="dash", line_color="gray", annotation_text="Ort. Satış Adedi")
            fig_portfolio.add_hline(y=ortalama_kar_marji, line_dash="dash", line_color="gray", annotation_text="Ort. Kar Marjı")
            st.plotly_chart(fig_portfolio, use_container_width=True)
            with st.expander("ℹ️ Gruplar Ne Anlama Geliyor?"):
                st.markdown("""
                - **Yıldızlar (Sağ Üst):** Hem çok satan hem de kar marjı yüksek olan, en değerli ürünleriniz.
                - **Nakit İnekleri (Sağ Alt):** Çok satılan ama kar marjı düşük olan, sürümden kazandıran ürünler.
                - **Soru İşaretleri (Sol Üst):** Az satılan ama kar marjı yüksek olan ürünler. Pazarlama ile yıldıza dönüşebilirler.
                - **Zayıflar (Sol Alt):** Hem az satan hem de az kazandıran ürünler. Stoktan çıkarmayı düşünebilirsiniz.
                """)
        else:
            st.info("Portföy analizi için yeterli veri bulunamadı.")
        
        st.markdown("---")
        st.subheader("En İyi 10 Ürün Görselleştirmesi")
        secilen_metrik = st.selectbox("Hangi metriğe göre sıralamak istersiniz?", 
                                      ('Toplam_Ciro', 'Toplam_Net_Kar', 'Benzersiz_Musteri_Sayisi', 'Toplam_Satis_Adedi'))
        top_10_urun = df_abc.nlargest(10, secilen_metrik)
        fig = px.bar(top_10_urun, x=secilen_metrik, y='UrunKodu', orientation='h',
                     title=f"En İyi 10 Ürün ({secilen_metrik.replace('_', ' ')})",
                     labels={'UrunKodu': 'Ürün Kodu'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# --- SEKME 2: PAZAR SEPETİ ANALİZİ ---
with tab2:
    st.header("Pazar Sepeti Analizi (Birliktelik)")
    st.markdown("Müşterilerin **aynı alışveriş sepeti içinde** hangi ürünleri birlikte satın alma eğiliminde olduğunu keşfedin.")

    col_param1, col_param2 = st.columns(2)
    with col_param1:
        min_support_degeri = st.slider("Analiz Hassasiyeti (Minimum Destek)", 0.01, 0.1, 0.05, 0.01)
    with col_param2:
        max_urun_sayisi = st.number_input("Analize dahil edilecek en popüler ürün sayısı", 
                                          min_value=50, max_value=1000, value=300, step=50)

    if st.button("Birliktelik Analizini Çalıştır", type="primary"):
        with st.spinner('Birliktelik kuralları hesaplanıyor...'):
            kurallar_df = market_basket_analizi_yap(temiz_df, min_support=min_support_degeri, max_urun_sayisi=max_urun_sayisi)
        st.session_state['birliktelik_kurallari'] = kurallar_df

    if 'birliktelik_kurallari' in st.session_state:
        kurallar_df = st.session_state['birliktelik_kurallari']

        if kurallar_df.empty:
            st.warning("Bu parametrelerle anlamlı birliktelik kuralı bulunamadı. Lütfen 'Minimum Destek' değerini düşürerek tekrar deneyin.")
        else:
            st.success(f"Analiz tamamlandı! Toplam {len(kurallar_df)} adet birliktelik kuralı bulundu.")
            st.markdown("---")

            st.subheader("🎁 Ürün Paketi Öneri Aracı")
            
            if not kurallar_df.empty:
                antecedents_flat = kurallar_df['antecedents'].explode()
                consequents_flat = kurallar_df['consequents'].explode()
                tum_urunler_listesi = pd.concat([antecedents_flat, consequents_flat]).unique()
                secilen_urun_paket = st.selectbox("Bir ürün seçin, size en iyi paket önerisini sunalım:", sorted(tum_urunler_listesi))

                if secilen_urun_paket:
                    oneriler = kurallar_df[kurallar_df['antecedents'].apply(lambda x: secilen_urun_paket in x)].sort_values('confidence', ascending=False)
                    if oneriler.empty:
                        st.info(f"'{secilen_urun_paket}' için doğrudan bir paket önerisi bulunamadı.")
                    else:
                        st.markdown(f"**'{secilen_urun_paket}'** alan müşterilere önerebileceğiniz en iyi ürünler:")
                        oneriler['consequents_str'] = oneriler['consequents'].apply(lambda x: ', '.join(list(x)))
                        for _, row in oneriler.head(3).iterrows():
                            st.success(f"**Öneri:** {row['consequents_str']} (Güven Oranı: {row['confidence']:.1%})")
            
            st.markdown("---")

            st.subheader("🌐 Birliktelik Ağı Grafiği")
            kural_sayisi = st.slider("Görselleştirilecek en güçlü kural sayısı:", 10, min(100, len(kurallar_df)), 25, 5)

            df_graph = kurallar_df[
                (kurallar_df['antecedents'].apply(len) == 1) & 
                (kurallar_df['consequents'].apply(len) == 1)
            ].nlargest(kural_sayisi, 'lift')

            if not df_graph.empty:
                G = nx.DiGraph()
                for _, row in df_graph.iterrows():
                    antecedent = list(row['antecedents'])[0]
                    consequent = list(row['consequents'])[0]
                    G.add_edge(antecedent, consequent, weight=row['lift'])

                pos = nx.spring_layout(G, k=0.5, iterations=50)

                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text, textposition="top center",
                                        marker=dict(showscale=True, colorscale='YlGnBu', size=10, colorbar=dict(thickness=15, title='Node Connections')))
                
                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(title='Ürün Birliktelik Ağı (Oklar -> yönünü gösterir)', showlegend=False,
                                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bu parametrelerle bir ağ grafiği oluşturulamadı. (Tekli ürün içeren kural bulunamadı)")

            with st.expander("Tüm Birliktelik Kurallarını Görüntüle"):
                st.dataframe(kurallar_df.style.format({'support': '{:.2%}', 'confidence': '{:.2%}', 'lift': '{:.2f}'}))

# --- SEKME 3: ÜRÜN-SEGMENT PROFİLİ ---
with tab3:
    st.header("👥 Ürün-Segment Profili Analizi")
    st.markdown("Hangi ürünlerin hangi müşteri segmentleri tarafından daha çok tercih edildiğini keşfedin.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bir Ürünün Müşteri Profili")
        urun_listesi = sorted(temiz_df['UrunKodu'].unique())
        secilen_urun = st.selectbox("Analiz edilecek bir ürün seçin:", urun_listesi)
        
        if secilen_urun:
            segment_dagilimi = urun_icin_segment_profili(temiz_df, sonuclar_df, secilen_urun)
            
            if segment_dagilimi.empty:
                st.info("Bu ürün için segment bilgisi bulunamadı.")
            else:
                fig_pie = px.pie(values=segment_dagilimi.values, names=segment_dagilimi.index, 
                                 title=f"'{secilen_urun}' Ürününü Satın Alanların Segment Dağılımı",
                                 color_discrete_sequence=px.colors.sequential.Plasma_r)
                st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Bir Segmentin Popüler Ürünleri")
        segment_listesi = sorted(sonuclar_df['Segment'].unique())
        secilen_segment = st.selectbox("Analiz edilecek bir segment seçin:", segment_listesi)
        
        if secilen_segment:
            populer_urunler = segment_icin_urun_profili(temiz_df, sonuclar_df, secilen_segment)
            
            if populer_urunler.empty:
                st.info("Bu segment için popüler ürün bilgisi bulunamadı.")
            else:
                fig_bar = px.bar(populer_urunler, y=populer_urunler.index, x=populer_urunler.values,
                                 orientation='h', title=f"'{secilen_segment}' Segmentinin En Çok Aldığı 10 Ürün (Ciroya Göre)",
                                 labels={'y': 'Ürün Kodu', 'x': 'Toplam Ciro (€)'})
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)