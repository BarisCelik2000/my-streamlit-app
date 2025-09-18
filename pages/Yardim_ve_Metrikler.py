# pages/Yardım_ve_Metrikler.py
# SORUMLULUĞU: Uygulamadaki analizleri ve metrikleri açıklayan bir rehber sunmak.

import streamlit as st
from data_handler import veriyi_yukle_ve_temizle

st.set_page_config(page_title="Yardım ve Metrikler", layout="wide")

@st.cache_data
def veriyi_getir():
    dosya_adi = 'satis_verileri_guncellenmis.json'
    return veriyi_yukle_ve_temizle(dosya_adi)

st.title("ℹ️ Yardım ve Metrik Tanımları")

st.markdown("""
Bu sayfa, dashboard'da kullanılan analizlerin ve metriklerin ne anlama geldiğini açıklamaktadır. 
Amacı, teknik bilgisi olmayan kullanıcıların da dashboard'dan en yüksek verimi almasını sağlamaktır.
""")

temiz_df = veriyi_getir()
son_guncelleme = temiz_df['Tarih'].max().strftime('%d-%m-%Y')
st.info(f"Kullanılan verinin son kayıt tarihi: **{son_guncelleme}**")

st.markdown("---")

# --- Metrik Açıklamaları ---
st.header("📈 Analizler ve Metrikler")

with st.expander("RFM Analizi ve Müşteri Performans Skoru (MPS)"):
    st.subheader("RFM Nedir?")
    st.markdown("""
    RFM, müşteri değerini ve davranışını ölçmek için kullanılan bir pazarlama analiz modelidir. Üç temel metriğe dayanır:
    - **Recency (Yenilik):** Müşterinin son alışverişinden bu yana geçen gün sayısıdır. **Düşük olması daha iyidir.**
    - **Frequency (Sıklık):** Müşterinin toplam alışveriş (işlem) sayısıdır. **Yüksek olması daha iyidir.**
    - **Monetary (Parasal Değer):** Müşterinin toplam harcama tutarıdır. **Yüksek olması daha iyidir.**
    """)
    st.subheader("Müşteri Performans Skoru (MPS)")
    st.markdown("MPS, her bir müşteri için hesaplanan R, F ve M skorlarının ağırlıklı bir ortalamasıdır. 0-100 arasında bir değer alır ve müşterinin şirkete olan genel değerini tek bir metrikle özetler. **Yüksek MPS, daha değerli bir müşteri anlamına gelir.**")

with st.expander("Müşteri Segmentleri"):
    st.markdown("""
    Müşteriler, MPS skorlarına göre 5 ana gruba ayrılır. Bu, pazarlama ve iletişim stratejilerini kişiselleştirmeyi kolaylaştırır.
    - **Şampiyonlar:** En iyi müşterileriniz. Sadakat programları ve özel tekliflerle ödüllendirilmelidirler.
    - **Potansiyel Şampiyonlar:** Şampiyon olma potansiyeli taşıyan, sadık ve değerli müşteriler. Yeni ürünler ve özel kampanyalarla desteklenmelidirler.
    - **Sadık Müşteriler:** Düzenli olarak alışveriş yapan ancak harcama potansiyelleri daha düşük olabilen grup.
    - **Riskli Müşteriler:** Eskiden iyi olan ancak son zamanlarda alışveriş sıklığı veya harcaması düşen müşteriler. Geri kazanma kampanyaları için ideal hedeflerdir.
    - **Kayıp Müşteriler:** Uzun süredir alışveriş yapmayan ve kaybedilmiş olarak kabul edilen müşteriler.
    """)

with st.expander("Müşteri Yaşam Boyu Değeri (CLV - Customer Lifetime Value)"):
    st.markdown("""
    CLV, bir müşterinin şirketinizle olan ilişkisi boyunca size getireceği **tahmini net karı** ifade eder. Geçmiş harcamaları, satın alma sıklığı ve genel churn (kayıp) oranı gibi faktörlere dayanarak hesaplanır. Pazarlama bütçenizi en değerli müşterilere yönlendirmenize yardımcı olan en stratejik metriklerden biridir.
    """)

with st.expander("Churn (Müşteri Kaybı) ve SHAP Analizi"):
    st.markdown("""
    **Churn Olasılığı:** Gelişmiş bir makine öğrenmesi modeli (`Random Forest`) tarafından, her bir müşterinin RFM değerlerine bakılarak hesaplanan bir olasılık skorudur. Bu skor, müşterinin yakın gelecekte sizi terk etme riskini yüzde olarak ifade eder.
    **SHAP Değeri Nedir?** Churn Neden Analizi sayfasında kullanılan SHAP, bir modelin kararını açıklamak için kullanılan en modern yöntemlerden biridir. Bir müşterinin churn olasılığının neden yüksek (veya düşük) olduğunu, her bir faktörün (Recency, Frequency, Monetary) bu karara ne kadar etki ettiğini **sayısal olarak** gösterir. Bu, modelin "içini görmemizi" sağlar.
    """)

with st.expander("Anomali Tespiti"):
    st.markdown("""
    Bu analiz, normalin dışında davranış gösteren müşteri ve işlemleri tespit eder.
    - **Genel Profil Anomalisi:** Bir müşterinin RFM profilinin, genel müşteri kitlesinin normal davranış kalıplarından ne kadar saptığını gösterir.
    - **Davranışsal Anomali:** Bir müşterinin **kendi normal satın alma ritminin** dışına çıktığı anları tespit eden bir erken uyarı sistemidir.
    - **İşlem Bazlı Anomali:** Tekil işlem bazında aykırı durumları (örn: sahtekarlık şüphesi, çok büyük bir sipariş) tespit eder.
    """)
    
with st.expander("Müşteri Benzerlik Analizi (Look-alike)"):
    st.markdown("""
    Bu analiz, seçtiğiniz bir "kaynak" müşterinin veya segmentin davranışsal profiline en çok benzeyen diğer müşterileri bulur. Bu yöntem, pazarlama kampanyalarınız için yeni ve potansiyeli yüksek hedef kitleler oluşturmak için kullanılır. Analiz, hem RFM (davranış) hem de satın alınan ürünler (zevk) bazında benzerliği hesaba katabilir.
    """)

st.markdown("---")

st.header("📑 Sayfalar ve Kullanım Amaçları")
st.markdown("Tüm sayfa açıklamaları, eklenen yeni özelliklere göre güncellenmiştir.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Genel Bakış")
    st.write("Uygulamanın ana sayfasıdır. Genel performansı özetleyen KPI'ları, grafikleri ve en değerli/riskli müşteri listelerini sunar. Anomali tespiti yapılan müşteriler ⚠️ ikonu ile işaretlenir.")

    st.subheader("📦 Ürün Analizi")
    st.write("Ürün performansını (ABC Analizi, Portföy Analizi), ürünlerin sepet birlikteliklerini (Ağ Grafiği, Paket Önerici) ve ürün-segment ilişkilerini inceler.")

    st.subheader("👤 Müşteri Detayı")
    st.write("Tek bir müşterinin 360 derecelik bir görünümünü sunar. Geçmiş davranış özeti, segment ortalamasına göre kıyaslama, segment yolculuğu, satış tahmini ve kişisel ürün önerilerini içerir.")

    st.subheader("📈 Kohort Analizi")
    st.write("Müşteri elde tutma oranını (retention) analiz eder. Artık farklı metrikler (müşteri sayısı, ortalama harcama) ve farklı zaman aralıkları (aylık, çeyreklik) için analiz yapabilir.")

    st.subheader("🗺️ Müşteri Yaşam Döngüsü")
    st.write("Müşterilerin **kazanım, segmentler arası geçiş ve kayıp (churn)** süreçlerini içeren tam yaşam döngüsünü interaktif bir Sankey diyagramı ile gösterir. Hem müşteri sayısı hem de CLV bazında analiz imkanı sunar.")

    st.subheader("🔮 Gelişmiş Tahminleme")
    st.write("Farklı modellerle (Prophet, ARIMA, Random Forest vb.) şirket geneli için satış tahmini yapar. 'Otomatik En İyi Model' seçeneği, güven aralıkları ve interaktif 'Senaryo Planlama' aracı içerir.")

    st.subheader("📉 Churn Neden Analizi")
    st.write("Daha güçlü bir model (Random Forest) ve daha güvenilir bir açıklama yöntemi (SHAP) kullanarak müşteri kaybının arkasındaki nedenleri inceler. Analizi segmente özel yapma imkanı sunar.")

with col2:
    st.subheader("🎯 Pazarlama ve Kampanya")
    st.write("Segmentlere özel kampanya fikirleri, potansiyel bir kampanyanın finansal getirisini ölçen ROI Simülatörü ve en karlı indirim oranını bulan Optimizasyon Aracı içerir.")

    st.subheader("⚠️ Anomali Tespiti")
    st.write("Profil, davranış ve işlem bazında aykırı durumları tespit eder. Artık anomalilerin nedenlerini açıklar ve skor bazında önceliklendirme imkanı sunar.")

    st.subheader("👥 Müşteri Benzerlik Analizi")
    st.write("Tek bir müşteriye veya bütün bir segmente benzeyen yeni hedef kitleler (look-alike) oluşturur. Benzerliği hem davranış (RFM) hem de ürün zevkine göre hesaplayabilir.")

    st.subheader("🔬 Segmentasyon Laboratuvarı")
    st.write("Farklı algoritma ve metrikler (`CLV`, `Churn Olasılığı` vb.) kullanarak özel müşteri segmentasyonları yaratmanızı sağlayan bir deney platformudur. Oluşturulan kümelere otomatik olarak 'persona' isimleri atar.")

    st.subheader("📊 Karşılaştırma Araçları")
    st.write("Müşterileri, segmentleri ve zaman periyotlarını karşılaştırır. Segmentlerin metrik dağılımlarını (kutu grafiği) ve dönemler arası müşteri değeri göçünü (Sankey) içerir.")

    st.subheader("🔀 Çapraz Kategori Analizi")
    st.write("Kategorilerin performansını, yaşam döngüsünü, sepet birlikteliklerini (Ağ Grafiği), müşteri profillerini ve kannibalizasyon risklerini analiz eder. Ayrıca 'Bir Sonraki Kategori' öneri motoru içerir.")
    
    st.subheader("ℹ️ Yardım ve Metrikler")
    st.write("Şu an bulunduğunuz bu sayfa, uygulamadaki tüm analizlerin ve metriklerin güncel açıklamalarını içerir.")