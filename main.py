# main.py

from data_handler import veriyi_yukle_ve_temizle, musteri_zaman_serisi_hazirla
from analysis_engine import rfm_skorlarini_hesapla, musterileri_segmentle, satis_tahmini_yap, tahmin_grafigini_ciz
from analysis_engine import (rfm_skorlarini_hesapla, 
                           musterileri_segmentle, 
                           satis_tahmini_yap, 
                           tahmin_grafigini_ciz,
                           churn_tahmin_modeli_olustur,
                           clv_hesapla)

import os
from data_handler import veriyi_yukle_ve_temizle, musteri_zaman_serisi_hazirla

def main():
    """
    Ana iş akışını yöneten fonksiyon.
    """
    print("Müşteri Analitik Projesi Başlatıldı...")
    
    # 1. Veri Yükleme ve Temizleme
    temiz_df = veriyi_yukle_ve_temizle('satis_verileri.xlsx')
    
    # 2. RFM Analizi ve Segmentasyon
    rfm_sonuclari = rfm_skorlarini_hesapla(temiz_df)
    segmentli_sonuclar = musterileri_segmentle(rfm_sonuclari)
    
    print("\n" + "="*50)
    print("  MÜŞTERİ SEGMENTASYON SONUÇLARI")
    print("="*50)
    print(segmentli_sonuclar.head(10))
    
    # 3. Müşteri Bazlı Satış Tahminlemesi (Çoklu Müşteri İçin)
    print("\n" + "="*50)
    print("  MÜŞTERİ BAZLI SATIŞ TAHMİNLEMESİ")
    print("="*50)

    # Ayarlar
    tahmin_yapilacak_musteri_sayisi = 3 # En iyi kaç şampiyon müşteri için tahmin yapılsın?
    tahmin_periyodu_ay = 6             # Kaç aylık tahmin yapılsın?

    # Tahmin yapmak için "Şampiyonlar" segmentindeki en iyi müşterileri seçelim
    sampiyonlar = segmentli_sonuclar[segmentli_sonuclar['Segment'] == 'Şampiyonlar']
    hedef_musteriler = sampiyonlar.head(tahmin_yapilacak_musteri_sayisi).index

    print(f"En iyi {len(hedef_musteriler)} şampiyon müşteri için {tahmin_periyodu_ay} aylık tahmin yapılacak...")

    # Seçilen her bir müşteri için döngü başlat
    for musteri_id in hedef_musteriler:
        print(f"\n--- Tahmin Başlatıldı: Müşteri ID -> {musteri_id} ---")
        
        # Seçilen müşteri için zaman serisi verisini hazırlayalım
        musteri_ts = musteri_zaman_serisi_hazirla(temiz_df, musteri_id)
        
        # Tahmin yapabilmek için yeterli veri olup olmadığını kontrol edelim (en az 1 yıl)
        if len(musteri_ts) >= 12:
            # Modeli çalıştırıp tahmin yapalım
            model, tahmin = satis_tahmini_yap(musteri_ts, ay_sayisi=tahmin_periyodu_ay)
            
            # Tahmin sonuçlarının son birkaç satırını (geleceği) gösterelim
            print(f"\n'{musteri_id}' için Tahmin Sonuçları (Gelecek {tahmin_periyodu_ay} Ay):")
            print(tahmin[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(tahmin_periyodu_ay))
            
            # Sonucu grafiğe dökelim (artık dosyaya kaydedecek)
            tahmin_grafigini_ciz(model, tahmin, musteri_id=musteri_id)
        else:
            print(f"'{musteri_id}' için yeterli geçmiş veri bulunamadı (en az 12 ay gerekli). Tahmin yapılamadı.")

    # 4. Müşteri Kaybı (Churn) Tahmini
    print("\n" + "="*50)
    print("  MÜŞTERİ KAYBI (CHURN) TAHMİNİ")
    print("="*50)
    
    churn_sonuclari, model_dogrulugu = churn_tahmin_modeli_olustur(segmentli_sonuclar)
    
    if model_dogrulugu > 0:
        print("En Yüksek Churn Riski Taşıyan 15 Müşteri:")
        # Sadece ilgili kolonları ve en riskli müşterileri gösterelim
        print(churn_sonuclari[['Recency', 'Frequency', 'Segment', 'Churn_Olasiligi']].head(15))

    
    # 5. Müşteri Yaşam Boyu Değeri (CLV) Hesaplama
    print("\n" + "="*50)
    print("  MÜŞTERİ YAŞAM BOYU DEĞERİ (CLV) HESAPLAMA")
    print("="*50)

    # Churn sonuçlarını içeren en güncel dataframe'i kullanarak CLV'yi hesapla
    clv_sonuclari = clv_hesapla(churn_sonuclari, kar_marji=0.25)
    
    print("\nYaşam Boyu Değeri En Yüksek 15 Müşteri:")
    # İlgili CLV kolonlarını ve en değerli müşterileri gösterelim
    print(clv_sonuclari[['Segment', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi']].head(15))

    print("\nProje başarıyla tamamlandı.")

if __name__ == "__main__":
    main()