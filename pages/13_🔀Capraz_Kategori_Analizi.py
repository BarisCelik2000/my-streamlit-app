# pages/13_Capraz_Kategori_Analizi.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import (rfm_skorlarini_hesapla, musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, clv_hesapla,
                           kategori_migrasyon_analizi_yap, kategori_performans_analizi_yap, 
                           kategori_kannibalizasyon_analizi, otomatik_kannibalizasyon_bul,
                           kategori_yasam_dongusu_analizi_yap, kategori_musteri_profili_analizi_yap,
                           kategori_sepet_birlikteligi_yap,
                           sonraki_kategori_onerisi)

st.set_page_config(page_title="Çapraz Kategori Analizi", layout="wide")

@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri.xlsx' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    sonuclar_df = clv_df 
    return temiz_df, sonuclar_df

temiz_df, sonuclar_df = veriyi_getir_ve_isle()

st.title("🔀 Çapraz Kategori Analizi")
st.markdown("Bu sayfa, müşterilerinizin ürün kategorileri arasındaki satın alma yolculuğunu ve kategorilerin kendi performanslarını analiz eder.")



# --- GÜNCELLENMİŞ SEKME İSİMLERİ ---
tab_names = [
    "Performans", "Migrasyon", "Kannibalizasyon", 
    "Yaşam Döngüsü", "Profiller", "Sepet Analizi", 
    "Yönlendirme"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

with tab1:
    st.header("Kategori Performans Panosu")
    
    # Veri dosyasında 'Kategori' sütunu olup olmadığını kontrol et
    if 'Kategori' not in temiz_df.columns:
        st.error("Performans analizi için 'Kategori' sütunu bulunamadı.")
    else:
        with st.spinner("Kategori performansları hesaplanıyor..."):
            # Analizi doğrudan bu sayfada yapabiliriz, çünkü analysis_engine'e taşımaya gerek yok
            performans_df = temiz_df.groupby('Kategori').agg(
                Toplam_Ciro=('ToplamTutar', 'sum'),
                Toplam_Net_Kar=('NetKar', 'sum'),
                Benzersiz_Musteri_Sayisi=('MusteriID', 'nunique'),
                Islem_Sayisi=('UrunKodu', 'count')
            ).reset_index()
            performans_df['Kar_Marji'] = (performans_df['Toplam_Net_Kar'] / performans_df['Toplam_Ciro']) * 100
            performans_df['Musteri_Basina_Ciro'] = performans_df['Toplam_Ciro'] / performans_df['Benzersiz_Musteri_Sayisi']
            performans_df.fillna(0, inplace=True)
            performans_df = performans_df.sort_values('Toplam_Ciro', ascending=False)
        
        if performans_df.empty:
            st.warning("Performans analizi için yeterli kategori verisi bulunamadı.")
        else:
            st.markdown("Tüm kategorilerinizin temel performans metrikleri:")
            st.dataframe(performans_df.style.format({
                'Toplam_Ciro': '{:,.0f} €', 'Toplam_Net_Kar': '{:,.0f} €',
                'Kar_Marji': '{:.1f}%', 'Musteri_Basina_Ciro': '{:,.2f} €'
            }))

with tab2:
    st.header("Kategori Migrasyon (Geçiş) Analizi")
    st.markdown("Müşterilerin ilk ve ikinci alışveriş kategorileri arasındaki geçişi gösterir.")
    
    if 'Kategori' not in temiz_df.columns:
        st.error("Migrasyon analizi için 'Kategori' sütunu bulunamadı.")
    else:
        if st.button("Migrasyon Analizini Çalıştır", type="primary"):
            with st.spinner("Kategori geçişleri analiz ediliyor..."):
                migrasyon_matrisi = kategori_migrasyon_analizi_yap(temiz_df)
            
            st.success("Analiz tamamlandı!")
            
            if migrasyon_matrisi.empty:
                st.warning("Kategori geçişi analizi için yeterli veri bulunamadı.")
            else:
                fig = px.imshow(migrasyon_matrisi, text_auto=".1%", aspect="auto",
                                labels=dict(x="İkinci Alım Kategorisi", y="İlk Alım Kategorisi", color="Geçiş Oranı"),
                                title="İlk Alımdan İkinci Alıma Kategori Geçiş Oranları (%)")
                fig.update_layout(margin=dict(t=80))
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔪 Kategori Kannibalizasyonu (Yamyamlık) Tespiti")
    
    # --- BÖLÜM 1: MANUEL ANALİZ ---
    st.subheader("Manuel Analiz")
    st.markdown("İki spesifik kategori arasındaki müşteri geçişini ve finansal etkisini inceleyin.")
    
    kategori_listesi = sorted(temiz_df['Kategori'].unique())
    col1, col2 = st.columns(2)
    with col1:
        kaynak_kategori = st.selectbox("Terk Edilen (Kaynak) Kategoriyi Seçin:", kategori_listesi, index=0)
    with col2:
        hedef_kategori = st.selectbox("Geçiş Yapılan (Hedef) Kategoriyi Seçin:", kategori_listesi, index=1 if len(kategori_listesi) > 1 else 0)            
    if st.button("Kannibalizasyon Analizini Çalıştır", type="primary"):
        if kaynak_kategori == hedef_kategori:
            st.error("Lütfen birbirinden farklı iki kategori seçin.")
        else:
            with st.spinner("Müşteri geçişleri ve finansal etki hesaplanıyor..."):
                gecis_yapan_df, sonuclar = kategori_kannibalizasyon_analizi(temiz_df, kaynak_kategori, hedef_kategori)
            
            st.success("Analiz tamamlandı!")
            
            if isinstance(sonuclar, str): # Eğer fonksiyon bir hata mesajı döndürdüyse
                st.warning(sonuclar)
            else:
                st.subheader(f"'{kaynak_kategori}' -> '{hedef_kategori}' Geçişinin Finansal Özeti")
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Geçiş Yapan Müşteri Sayısı", f"{sonuclar['Geçiş Yapan Müşteri Sayısı']:.0f}")
                kpi_cols[1].metric("Kaybedilen Ciro", f"{sonuclar['Kaybedilen Ciro (Kaynak Kategoriden)']:,.0f} €")
                kpi_cols[2].metric("Kazanılan Ciro", f"{sonuclar['Kazanılan Ciro (Hedef Kategoriden)']:,.0f} €")
                kpi_cols[3].metric("Net Ciro Etkisi", f"{sonuclar['Net Ciro Etkisi']:,.0f} €", delta_color="inverse")

                with st.expander("Geçiş Yapan Müşterilerin Listesini Görüntüle"):
                    # --- DÜZELTİLMİŞ BÖLÜM ---
                    # İsim bilgilerini ham veriden (temiz_df) alıp geçiş tablosuyla birleştiriyoruz.
                    if 'MusteriAdi' in temiz_df.columns:
                        isimler_df = temiz_df[['MusteriID', 'MusteriAdi']].drop_duplicates()
                        gosterilecek_df = pd.merge(gecis_yapan_df[['MusteriID']].drop_duplicates(), isimler_df, on='MusteriID', how='left')
                        st.dataframe(gosterilecek_df)
                    else:
                        # Eğer isim kolonu yoksa sadece ID'leri göster
                        st.dataframe(gecis_yapan_df[['MusteriID']].drop_duplicates())
    st.markdown("---")
    st.subheader("Otomatik Analiz")
    st.markdown("Sistemin, tüm kategoriler arasında en fazla müşteri geçişinin yaşandığı **'yamyamlık' potansiyeli en yüksek** noktaları otomatik olarak bulmasını sağlayın.")
    
    if st.button("En Yüksek Geçişleri Otomatik Bul", type="secondary"):
        with st.spinner("Tüm olası kategori çiftleri analiz ediliyor... Bu işlem biraz zaman alabilir."):
            otomatik_sonuclar_df = otomatik_kannibalizasyon_bul(temiz_df)
        
        st.success("Otomatik analiz tamamlandı!")
        
        if otomatik_sonuclar_df.empty:
            st.info("Kategoriler arasında anlamlı bir müşteri geçişi (kannibalizasyon) tespit edilmedi.")
        else:
            st.markdown("**En Fazla Müşteri Geçişi Yaşanan Kategori Çiftleri:**")
            st.dataframe(otomatik_sonuclar_df.style.format({
                'Geçiş Yapan Müşteri Sayısı': '{:.0f}',
                'Net Ciro Etkisi': '{:,.0f} €',
                'Kaybedilen Ciro (Kaynak Kategoriden)': '{:,.0f} €',
                'Kazanılan Ciro (Hedef Kategoriden)': '{:,.0f} €'
            }).background_gradient(cmap='Reds', subset=['Geçiş Yapan Müşteri Sayısı']))

with tab4:
    st.header("🌀 Kategori Performans Zaman Çizgisi")
    st.markdown("Seçtiğiniz kategorilerin ve metriklerin zaman içindeki aylık trendlerini karşılaştırarak yaşam döngülerini analiz edin.")
    
    with st.spinner("Kategorilerin aylık performansları hesaplanıyor..."):
        yasam_dongusu_df = kategori_yasam_dongusu_analizi_yap(temiz_df)
    
    if yasam_dongusu_df.empty:
        st.warning("Yaşam döngüsü analizi için yeterli veri bulunamadı.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            kategori_listesi = sorted(yasam_dongusu_df['Kategori'].unique())
            secilen_kategoriler = st.multiselect(
                "Karşılaştırmak istediğiniz kategorileri seçin:",
                options=kategori_listesi,
                default=kategori_listesi[:min(5, len(kategori_listesi))]
            )
        with col2:
            metrik_map = {
                "Toplam Ciro (€)": "ToplamCiro",
                "Benzersiz Müşteri Sayısı": "BenzersizMusteriSayisi",
                "Kar Marjı (%)": "KarMarji"
            }
            secilen_metrik_adi = st.selectbox("Görüntülenecek metriği seçin:", metrik_map.keys())
            secilen_metrik_kodu = metrik_map[secilen_metrik_adi]

        if secilen_kategoriler and secilen_metrik_adi:
            plot_df = yasam_dongusu_df[yasam_dongusu_df['Kategori'].isin(secilen_kategoriler)]
            
            fig = px.line(
                plot_df,
                x='Tarih',
                y=secilen_metrik_kodu,
                color='Kategori',
                markers=True,
                title=f"Seçilen Kategorilerin Aylık '{secilen_metrik_adi}' Trendi",
                labels={'Tarih': 'Ay', secilen_metrik_kodu: secilen_metrik_adi}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Bu grafik, hangi kategorilerin istikrarlı bir şekilde büyüdüğünü, hangilerinin mevsimsel olduğunu ve hangilerinin popülerliğini kaybettiğini seçtiğiniz metrik üzerinden görmenizi sağlar.")

with tab5:
    st.header("👥 Kategori Müşteri Profilleri")
    st.markdown("Her bir kategoriyi satın alan müşteri kitlesinin segmentlere göre dağılımını inceleyin.")
    
    with st.spinner("Kategorilerin müşteri profilleri oluşturuluyor..."):
        profil_df = kategori_musteri_profili_analizi_yap(temiz_df, sonuclar_df)
        
    if profil_df.empty:
        st.warning("Müşteri profili analizi için yeterli veri bulunamadı.")
    else:
        st.subheader("Her Kategorinin Müşteri Segmenti Dağılımı (%)")
        
        # Yığılmış bar grafiği (stacked bar chart) ile görselleştirme
        fig = px.bar(
            profil_df,
            barmode='stack',
            title="Kategorilerin Müşteri Profili Kompozisyonu",
            labels={'value': 'Müşteri Oranı (%)', 'Kategori': 'Ürün Kategorisi', 'variable': 'Müşteri Segmenti'},
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("""
        **Nasıl Yorumlanır?**
        - Her bir dikey bar, bir ürün kategorisini temsil eder.
        - Barın içindeki her bir renkli dilim, o kategoriyi satın alan müşterilerin yüzde kaçının hangi segmente ait olduğunu gösterir.
        - Örneğin, "Premium Ürünler" kategorisinin barında büyük bir mor dilim varsa, bu kategorinin ağırlıklı olarak "Şampiyon" müşteriler tarafından tercih edildiğini anlarız.
        - Bu grafik, hangi segmentlere hangi kategorileri pazarlamanız gerektiği konusunda size yol gösterir.
        """)
        
        with st.expander("Detaylı Veri Tablosunu Görüntüle"):
            st.dataframe(profil_df.style.format("{:.1f}%"))

with tab6:
    st.header("🛒 Kategori Sepet Birlikteliği Analizi")
    st.markdown("Müşterilerin **aynı alışveriş sepeti içinde** hangi kategorileri birlikte satın alma eğiliminde olduğunu keşfedin.")
    
    min_support_degeri = st.slider("Analiz Hassasiyeti (Minimum Destek)", 0.005, 0.1, 0.01, 0.005, format="%.3f",
                                help="Bir kategori kombinasyonunun 'sık' kabul edilmesi için tüm sepetlerin en az yüzde kaçında görünmesi gerektiğini belirtir.")

    if st.button("Sepet Birlikteliğini Analiz Et", type="primary"):
        with st.spinner("Tüm alışveriş sepetleri analiz ediliyor..."):
            sepet_kurallari_df = kategori_sepet_birlikteligi_yap(temiz_df, min_support=min_support_degeri)
        st.session_state.sepet_kurallari_df = sepet_kurallari_df
        
    if 'sepet_kurallari_df' in st.session_state:
        sepet_kurallari_df = st.session_state.sepet_kurallari_df
        
        if sepet_kurallari_df.empty:
            st.warning("Belirtilen hassasiyet seviyesinde anlamlı bir birliktelik kuralı bulunamadı. Lütfen daha düşük bir minimum destek değeri deneyin.")
        else:
            st.success(f"Analiz tamamlandı! {len(sepet_kurallari_df)} adet anlamlı kural bulundu.")
            
            # frozenset'leri okunaklı metne çevir
            df_display = sepet_kurallari_df.copy()
            df_display['antecedents'] = df_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            df_display['consequents'] = df_display['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.subheader("Tespit Edilen Kategori Birliktelik Kuralları")
            st.dataframe(df_display[['antecedents', 'consequents', 'confidence', 'lift']].rename(columns={
                'antecedents': 'Eğer Bu Kategori(ler) Alınırsa',
                'consequents': 'O Zaman Bu Kategori(ler) de Alınır',
                'confidence': 'Güven (%)',
                'lift': 'Lift (Güç)'
            }).style.format({'Güven (%)': '{:.1%}', 'Lift (Güç)': '{:.2f}'}))
            
            # --- YENİ EKLENEN BÖLÜM: Ağ Grafiği ---
            st.markdown("---")
            st.subheader("Kategori Birliktelik Ağı Grafiği")
            
            kural_sayisi = st.slider("Görselleştirilecek en güçlü kural sayısı:", 5, min(100, len(sepet_kurallari_df)), 20, 5)

            # Sadece tekli ilişkileri (1 kategori -> 1 kategori) görselleştirelim
            df_graph = sepet_kurallari_df[
                (sepet_kurallari_df['antecedents'].apply(len) == 1) & 
                (sepet_kurallari_df['consequents'].apply(len) == 1)
            ].nlargest(kural_sayisi, 'lift').copy()

            if not df_graph.empty:
                # frozenset'leri string'e çevir
                df_graph['source'] = df_graph['antecedents'].apply(lambda x: list(x)[0])
                df_graph['target'] = df_graph['consequents'].apply(lambda x: list(x)[0])

                G = nx.from_pandas_edgelist(df_graph, source='source', target='target', edge_attr='lift', create_using=nx.DiGraph())
                pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

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
                    node_text.append(f"{node} ({len(list(G.neighbors(node)))})")
                
                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                                        hoverinfo='text',
                                        marker=dict(showscale=True, colorscale='YlGnBu', size=15, 
                                                    color=[len(list(G.neighbors(node))) for node in G.nodes()],
                                                    colorbar=dict(thickness=15, title='Bağlantı Sayısı')))

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(title='Kategorilerin Sepet Birliktelik Ağı (En Güçlü Bağlantılar)', showlegend=False,
                                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                st.plotly_chart(fig, use_container_width=True)
                st.info("Bu grafikte her bir daire bir kategoriyi temsil eder. Dairenin büyüklüğü ve rengi, o kategorinin ne kadar çok başka kategoriyle birlikte satın alındığını gösterir. Oklar, satın alma yönünü belirtir (Örn: A -> B, A'yı alanların B'yi de aldığını gösterir).")
            else:
                st.warning("Grafik oluşturmak için yeterli sayıda tekli ilişki kuralı bulunamadı.")

with tab7:
    st.header("🎯 'Bir Sonraki Kategori' Öneri Motoru")
    st.markdown("Müşterilerin ilk alışveriş verilerine dayanarak, onlara tanıtılması en mantıklı olan **ikinci kategoriyi** keşfedin.")
    st.info("Bu araç, `Kategori Migrasyon Analizi` verilerini kullanarak, bir başlangıç kategorisinden diğerlerine olan doğal müşteri akışını analiz eder.")

    # Migrasyon matrisini bu sekmeye özel olarak, cache'leyerek hesapla
    @st.cache_data
    def migrasyon_getir(_df):
        return kategori_migrasyon_analizi_yap(_df)

    migrasyon_matrisi = migrasyon_getir(temiz_df)
    
    if migrasyon_matrisi.empty:
        st.warning("Öneri üretmek için yeterli kategori geçiş verisi bulunamadı.")
    else:
        kaynak_kategori = st.selectbox(
            "Müşterinin ilk alışveriş yaptığı başlangıç kategorisini seçin:",
            options=migrasyon_matrisi.index
        )
        
        if kaynak_kategori:
            st.markdown("---")
            st.subheader(f"'{kaynak_kategori}' Kategorisinden Sonra Önerilenler:")
            
            oneriler = sonraki_kategori_onerisi(migrasyon_matrisi, kaynak_kategori)
            
            if oneriler.empty:
                st.info("Bu başlangıç kategorisi için bir sonraki kategori önerisi bulunmuyor.")
            else:
                # En iyi 3 öneriyi göster
                for i, (kategori, oran) in enumerate(oneriler.head(3).items()):
                    st.success(f"**#{i+1} Öneri:** `{kategori}` kategorisi. \n\n (Müşterilerin **%{oran*100:.1f}**'i ikinci alışverişlerinde bu kategoriye yönelmiştir.)")

