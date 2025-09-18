# data_handler.py
# SORUMLULUĞU: Veriyi okumak, temel temizlik ve formatlama işlemlerini yapmak.

import pandas as pd
import numpy as np

def veriyi_yukle_ve_temizle(dosya_yolu, varsayilan_kar_marji=0.25):
    """
    NİHAİ VERSİYON: Dosyadaki kolon adlarına göre kendini ayarlar. 'Maliyet' kolonu
    yoksa hata vermek yerine, varsayılan kar marjı ile çalışır.
    .json desteği en sağlam haliyle güncellendi.
    """
    print(f"'{dosya_yolu}' yükleniyor...")
    
    if dosya_yolu.endswith('.xlsx'):
        df = pd.read_excel(dosya_yolu)
    elif dosya_yolu.endswith('.csv'):
        try:
            df = pd.read_csv(dosya_yolu, sep=';', decimal=',')
        except Exception:
            df = pd.read_csv(dosya_yolu, sep=',', decimal='.')
            
    # --- YENİ ve SAĞLAM JSON OKUMA MANTIĞI ---
    elif dosya_yolu.endswith('.json'):
        try:
            # 1. Önce, her satırın tek bir JSON olduğu "JSON Lines" formatını dene.
            # "Trailing data" hatasının en yaygın sebebi budur.
            df = pd.read_json(
                dosya_yolu, 
                orient='records', 
                lines=True, 
                convert_dates=['Tarih', 'tarih', 'Date', 'invoicedate']
            )
            print("✓ JSON dosyası 'JSON Lines' formatında başarıyla okundu.")
        except ValueError:
            # 2. Eğer "JSON Lines" başarısız olursa, standart formatı (tek bir büyük liste) dene.
            print("UYARI: JSON Lines formatı okunamadı, standart JSON formatı deneniyor...")
            try:
                df = pd.read_json(
                    dosya_yolu, 
                    orient='records', 
                    convert_dates=['Tarih', 'tarih', 'Date', 'invoicedate']
                )
                print("✓ JSON dosyası standart formatta başarıyla okundu.")
            except Exception as e:
                # 3. Eğer ikisi de başarısız olursa, kullanıcıya bilgilendirici bir hata göster.
                raise ValueError(
                    "JSON dosyası okunamadı. Lütfen dosyanın yapısını kontrol edin. "
                    "Dosya ya tek bir büyük liste '[{...}, {...}]' şeklinde olmalı, "
                    "ya da her satırda tek bir JSON nesnesi '{...}' içermelidir (JSON Lines). "
                    f"Orijinal Hata: {e}"
                )
    # --- GÜNCELLEME SONU ---
        
    else:
        raise ValueError("Desteklenmeyen dosya formatı. Lütfen .xlsx, .csv veya .json kullanın.")

    print("Veri başarıyla yüklendi. Temizleme işlemleri başlıyor...")
    
    df.columns = df.columns.str.strip().str.lower()
    
    kolon_map = {
        'musteriid': ['musteriid', 'müşteri id', 'customer id'],
        'urunkodu': ['urunkodu', 'ürün kodu', 'stockcode', 'product id'],
        'tarih': ['tarih', 'siparis tarihi', 'date', 'invoicedate'],
        'miktar': ['miktar', 'quantity'],
        'birimfiyat': ['birimfiyat', 'fiyat', 'price', 'unitprice'],
        'maliyet': ['maliyet', 'purchase cost', 'cost', 'purchasecost'],
        'kategori': ['kategori', 'category'] 
    }
    
    for standart_ad, potansiyel_adlar in kolon_map.items():
        for ad in potansiyel_adlar:
            if ad in df.columns:
                df.rename(columns={ad: standart_ad}, inplace=True)
                break
    
    sayisal_kolonlar = ['birimfiyat', 'miktar', 'maliyet']
    for kolon in sayisal_kolonlar:
        if kolon in df.columns:
            df[kolon] = pd.to_numeric(df[kolon].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    if 'tarih' in df.columns:
        df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')

    gerekli_kolonlar_temel = ['musteriid', 'tarih', 'miktar', 'birimfiyat', 'urunkodu']
    df.dropna(subset=gerekli_kolonlar_temel, inplace=True)
    df = df[df['miktar'] > 0]
    df = df[df['birimfiyat'] > 0]

    if 'maliyet' in df.columns:
        df = df[df['maliyet'] > 0]

    df['toplamtutar'] = df['miktar'] * df['birimfiyat']
    
    if 'maliyet' in df.columns:
        df.dropna(subset=['maliyet'], inplace=True)
        df['netkar'] = df['toplamtutar'] - (df['miktar'] * df['maliyet'])
        print("✓ Gerçek maliyet verisi bulundu ve 'NetKar' kolonu bu veriye göre hesaplandı.")
    else:
        df['netkar'] = df['toplamtutar'] * varsayilan_kar_marji
        print(f"UYARI: Dosyada 'Maliyet' kolonu bulunamadı. 'NetKar' kolonu, %{varsayilan_kar_marji*100} varsayılan kar marjı ile hesaplandı.")

    final_rename_map = {
        'musteriid': 'MusteriID', 'musteriadi': 'MusteriAdi', 'urunkodu': 'UrunKodu', 
        'kategori': 'Kategori', 'tarih': 'Tarih', 'miktar': 'Miktar', 
        'birimfiyat': 'BirimFiyat', 'maliyet': 'Maliyet', 
        'toplamtutar': 'ToplamTutar', 'netkar': 'NetKar'
    }
    df.rename(columns={k: v for k, v in final_rename_map.items() if k in df.columns}, inplace=True)

    print("Veri temizleme tamamlandı.")
    return df

def musteri_zaman_serisi_hazirla(df, musteri_id):
    """Belirli bir müşterinin verisini aylık zaman serisi formatına dönüştürür."""
    musteri_df = df[df['MusteriID'] == musteri_id].copy()
    aylik_satislar = musteri_df.set_index('Tarih').resample('ME').agg({'ToplamTutar': 'sum'}).reset_index()
    aylik_satislar.rename(columns={'Tarih': 'ds', 'ToplamTutar': 'y'}, inplace=True)
    return aylik_satislar

def genel_satis_trendi_hazirla(df):
    """Tüm şirketin aylık satış trendini Prophet'e uygun hale getirir."""
    aylik_satislar = df.set_index('Tarih').resample('ME').agg({
        'ToplamTutar': 'sum',
        'MusteriID': 'nunique',
        'Miktar': 'sum'
    }).reset_index()
    
    aylik_satislar.rename(columns={
        'Tarih': 'ds', 
        'ToplamTutar': 'y',
        'MusteriID': 'musteri_sayisi',
        'Miktar': 'toplam_miktar'
    }, inplace=True)
    
    return aylik_satislar


def gecmis_kampanya_verisi_uret(sonuclar_df, temiz_df, musteri_sayisi=500):
    """
    Mevcut müşteri profillerine dayanarak, makine öğrenmesi modelini eğitmek için
    sahte ama gerçekçi bir kampanya geçmişi verisi üretir.
    """
    # Analiz için rastgele bir müşteri alt kümesi seç
    ornek_musteriler_df = sonuclar_df.sample(n=min(musteri_sayisi, len(sonuclar_df)), random_state=42)
    
    olasi_aksiyonlar = ['İndirim Teklifi', 'Çapraz Satış Önerisi', 'Yeni Ürün Tanıtımı']
    olasi_kanallar = ['E-posta', 'SMS']
    
    kampanya_gecmisi = []
    
    for index, musteri in ornek_musteriler_df.iterrows():
        # Her müşteriye rastgele bir kampanya ata
        aksiyon = np.random.choice(olasi_aksiyonlar)
        kanal = np.random.choice(olasi_kanallar)
        
        # Dönüşüm olasılığını müşterinin özelliklerine göre belirle (simülasyonun kalbi)
        donusum_olasiligi = 1 - musteri['Churn_Olasiligi'] # Temel olasılık
        
        # Aksiyon ve segment uyumuna göre olasılığı ayarla
        if aksiyon == 'İndirim Teklifi' and musteri['Segment'] in ['Riskli Müşteriler', 'Kayıp Müşteriler']:
            donusum_olasiligi *= 1.5
        elif aksiyon == 'Çapraz Satış Önerisi' and musteri['Segment'] in ['Şampiyonlar', 'Potansiyel Şampiyonlar']:
            donusum_olasiligi *= 1.3
        
        # Olasılık 1'i geçemez
        donusum_olasiligi = min(donusum_olasiligi, 0.95)
        
        # Bu olasılığa göre müşterinin dönüşüm yapıp yapmadığını belirle
        donusum_yapti_mi = 1 if np.random.rand() < donusum_olasiligi else 0
        
        kampanya_gecmisi.append({
            'MusteriID': index,
            'Segment': musteri['Segment'],
            'Recency': musteri['Recency'],
            'Frequency': musteri['Frequency'],
            'Monetary': musteri['Monetary'],
            'SunulanAksiyon': aksiyon,
            'KullanilanKanal': kanal,
            'DonusumYaptiMi': donusum_yapti_mi
        })
        
    return pd.DataFrame(kampanya_gecmisi)

