# pages/Gelişmiş_Tahminleme.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_handler import veriyi_yukle_ve_temizle, genel_satis_trendi_hazirla
from analysis_engine import prophet_tahmin, arima_tahmin, sarima_tahmin, random_forest_tahmin, ensemble_tahmin, what_if_analizi

st.set_page_config(page_title="Gelişmiş Tahminleme", layout="wide")

@st.cache_data
def veriyi_getir():
    dosya_adi = 'satis_verileri.json'
    return veriyi_yukle_ve_temizle(dosya_adi)

st.title("🔮 Gelişmiş Tahminleme ve Senaryolar")
st.markdown("Bu sayfada farklı tahminleme modelleri kullanarak genel satış projeksiyonları yapabilir ve çeşitli iş senaryolarının potansiyel etkilerini test edebilirsiniz.")

temiz_df = veriyi_getir()

aylik_satislar = genel_satis_trendi_hazirla(temiz_df)

tab1, tab2, tab3 = st.tabs(["Genel Satış Tahmini", "Ensemble Tahmin", "What-if Analizi"])

with tab1:
    st.header("📊 Genel Satış Tahmini")
    
    st.subheader("Otomatik Tahmin (Önerilen)")
    tahmin_periyodu_oto = st.slider("Tahmin Periyodu (Ay)", 3, 24, 12, key="slider_oto")
    
    if st.button("En İyi Modeli Bul ve Tahmin Et", type="primary", use_container_width=True):
        if len(aylik_satislar) < 18:
            st.error("Otomatik model karşılaştırması için en az 18 aylık veri gerekmektedir.")
        else:
            with st.spinner("Tüm modeller test ediliyor ve en iyisi seçiliyor..."):
                metrikler = {}
                train_size = len(aylik_satislar) - 6
                train, test = aylik_satislar[:train_size], aylik_satislar[train_size:]
                
                _, forecast_p = prophet_tahmin(train.copy(), 6)
                metrikler['Prophet'] = np.sqrt(mean_squared_error(test['y'], forecast_p['yhat'].tail(6)))
                _, forecast_a, _ = arima_tahmin(train.copy(), 6)
                metrikler['ARIMA'] = np.sqrt(mean_squared_error(test['y'], forecast_a['tahmin']))
                _, forecast_s = sarima_tahmin(train.copy(), 6)
                if forecast_s is not None:
                    metrikler['SARIMA'] = np.sqrt(mean_squared_error(test['y'], forecast_s['tahmin']))
                _, forecast_rf = random_forest_tahmin(train.copy(), 6)
                if forecast_rf is not None:
                    metrikler['Random Forest'] = np.sqrt(mean_squared_error(test['y'], forecast_rf['tahmin']))
                
                en_iyi_model_adi = min(metrikler, key=metrikler.get)
                st.session_state.en_iyi_model = en_iyi_model_adi
                st.session_state.metrikler = metrikler

                if en_iyi_model_adi == 'Prophet':
                    _, forecast = prophet_tahmin(aylik_satislar.copy(), tahmin_periyodu_oto)
                elif en_iyi_model_adi == 'ARIMA':
                    _, forecast, _ = arima_tahmin(aylik_satislar.copy(), tahmin_periyodu_oto)
                elif en_iyi_model_adi == 'SARIMA':
                    _, forecast = sarima_tahmin(aylik_satislar.copy(), tahmin_periyodu_oto)
                elif en_iyi_model_adi == 'Random Forest':
                    _, forecast = random_forest_tahmin(aylik_satislar.copy(), tahmin_periyodu_oto)
                
                st.session_state.oto_tahmin_sonucu = forecast
                    
    if 'oto_tahmin_sonucu' in st.session_state:
        st.success(f"Analiz tamamlandı! En iyi performans gösteren model (en düşük hata ile): **{st.session_state.en_iyi_model}**")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aylik_satislar['ds'], y=aylik_satislar['y'], mode='lines+markers', name='Gerçek Veri'))
        
        sonuc = st.session_state.oto_tahmin_sonucu
        model_adi = st.session_state.en_iyi_model

        if model_adi == 'Prophet':
            future_data = sonuc.tail(tahmin_periyodu_oto)
            fig.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_lower'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', showlegend=False))
            fig.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Güven Aralığı'))
            fig.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat'], mode='lines', name=f'{model_adi} Tahmini', line=dict(color='green', dash='dash')))
        
        elif model_adi in ['ARIMA', 'SARIMA']:
            future_dates = sonuc.index
            fig.add_trace(go.Scatter(x=future_dates, y=sonuc['alt_sinir'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', showlegend=False))
            fig.add_trace(go.Scatter(x=future_dates, y=sonuc['ust_sinir'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Güven Aralığı'))
            fig.add_trace(go.Scatter(x=future_dates, y=sonuc['tahmin'], mode='lines', name=f'{model_adi} Tahmini', line=dict(color='green', dash='dash')))
        
        else: 
            future_dates = sonuc.index
            fig.add_trace(go.Scatter(x=future_dates, y=sonuc['tahmin'], mode='lines', name=f'{model_adi} Tahmini', line=dict(color='green', dash='dash')))

        fig.update_layout(title=f"Otomatik Seçilen En İyi Modele Göre Satış Tahmini", xaxis_title="Tarih", yaxis_title="Satış Tutarı (€)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Model Performans Metriklerini (RMSE) Görüntüle"):
            st.info("RMSE (Kök Ortalama Kare Hata), modelin tahminlerinin ortalama ne kadar saptığını gösterir. **Düşük olması daha iyidir.**")
            st.dataframe(pd.Series(st.session_state.metrikler, name="RMSE Değeri").reset_index().rename(columns={'index': 'Model'}))

    st.markdown("---")
    
    with st.expander("Gelişmiş Ayarlar ve Manuel Karşılaştırma"):
        tahmin_periyodu_manual = st.slider("Tahmin Periyodu (Ay)", 3, 24, 6, key="slider_manual")
        use_prophet = st.checkbox("Prophet", value=True)
        use_arima = st.checkbox("ARIMA", value=True)
        use_sarima = st.checkbox("SARIMA", value=True)
        use_rf = st.checkbox("Random Forest")
        
        if st.button("Seçili Modelleri Karşılaştır", key="model_calistir_manual"):
            with st.spinner("Seçili modeller eğitiliyor..."):
                sonuclar = {}
                if use_prophet:
                    _, forecast = prophet_tahmin(aylik_satislar.copy(), tahmin_periyodu_manual)
                    sonuclar['Prophet'] = forecast
                if use_arima:
                    _, forecast, _ = arima_tahmin(aylik_satislar.copy(), tahmin_periyodu_manual)
                    sonuclar['ARIMA'] = forecast
                if use_sarima:
                    _, forecast = sarima_tahmin(aylik_satislar.copy(), tahmin_periyodu_manual)
                    if forecast is not None: sonuclar['SARIMA'] = forecast
                if use_rf:
                    _, forecast = random_forest_tahmin(aylik_satislar.copy(), tahmin_periyodu_manual)
                    if forecast is not None: sonuclar['Random Forest'] = forecast
                st.session_state['tahmin_sonuclari_manual'] = sonuclar

        if 'tahmin_sonuclari_manual' in st.session_state:
            fig_manual = go.Figure()
            fig_manual.add_trace(go.Scatter(x=aylik_satislar['ds'], y=aylik_satislar['y'], mode='lines+markers', name='Gerçek Veri'))
            
            # --- DÜZELTİLMİŞ BÖLÜM ---
            colors = {'Prophet': '#636EFA', 'ARIMA': '#EF553B', 'SARIMA': '#FFA15A', 'Random Forest': '#AB63FA'}
            
            for name, result in st.session_state['tahmin_sonuclari_manual'].items():
                color_hex = colors.get(name, '#CCCCCC')
                color_rgba = f"rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, 0.2)"

                if name == 'Prophet':
                    future_data = result.tail(tahmin_periyodu_manual)
                    fig_manual.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_lower'], fill=None, mode='lines', line_color=color_rgba, showlegend=False))
                    fig_manual.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_upper'], fill='tonexty', mode='lines', line_color=color_rgba, name=f'{name} Güven Aralığı'))
                    fig_manual.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat'], mode='lines', name=f'{name} Tahmini', line=dict(color=color_hex, dash='dash')))
                
                elif name in ['ARIMA', 'SARIMA']:
                    future_dates = result.index
                    fig_manual.add_trace(go.Scatter(x=future_dates, y=result['alt_sinir'], fill=None, mode='lines', line_color=color_rgba, showlegend=False))
                    fig_manual.add_trace(go.Scatter(x=future_dates, y=result['ust_sinir'], fill='tonexty', mode='lines', line_color=color_rgba, name=f'{name} Güven Aralığı'))
                    fig_manual.add_trace(go.Scatter(x=future_dates, y=result['tahmin'], mode='lines', name=f'{name} Tahmini', line=dict(color=color_hex, dash='dash')))

                elif name == 'Random Forest':
                    future_dates = result.index
                    fig_manual.add_trace(go.Scatter(x=future_dates, y=result['tahmin'], mode='lines', name=f'{name} Tahmini', line=dict(color=color_hex, dash='dash')))
            
            fig_manual.update_layout(title="Manuel Model Karşılaştırması", xaxis_title="Tarih", yaxis_title="Satış Tutarı (€)")
            st.plotly_chart(fig_manual, use_container_width=True)

with tab2:
    st.header("🤖 Ensemble (Ortalama) Tahmin")
    aylik_satislar = genel_satis_trendi_hazirla(temiz_df)
    
    if st.button("Ensemble Tahmini Çalıştır", key="ensemble_calistir"):
        with st.spinner("Ensemble modeli çalıştırılıyor..."):
            ensemble_sonuclari = ensemble_tahmin(aylik_satislar.copy(), 12)
            st.session_state['ensemble_sonuclari'] = ensemble_sonuclari

    if 'ensemble_sonuclari' in st.session_state:
        st.subheader("Aylık Tahmin Dağılımı")
        df_ensemble = pd.DataFrame(st.session_state['ensemble_sonuclari'])
        df_ensemble.index = pd.date_range(start=aylik_satislar['ds'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='ME')
        st.dataframe(df_ensemble.style.format("{:,.0f} €").background_gradient(cmap='Greens', subset=['Ensemble']))

with tab3:
    st.header("💡 Senaryo Planlama (Prophet ile)")
    st.markdown("Geleceğe yönelik iş hedeflerinizi ve beklentilerinizi modele girerek ciro üzerindeki potansiyel etkisini simüle edin.")

    senaryo_periyodu = st.slider("Senaryo Periyodu (Ay)", 3, 24, 12, key="slider_senaryo")
    
    st.subheader("Gelecek Dönem Varsayımlarınızı Girin")
    st.caption("Bu ayarlar, Prophet modelinin harici regresörlerini besleyecektir. Değiştirmeden bırakırsanız, model geçmiş verilerin ortalamasını kullanacaktır.")

    # Tarihsel ortalamaları hesaplayıp varsayılan değer olarak kullanalım
    ort_musteri_sayisi = int(aylik_satislar['musteri_sayisi'].mean())
    ort_satis_adedi = int(aylik_satislar['toplam_miktar'].mean())

    gelecek_musteri_sayisi = st.slider(
        "Gelecek Aylık Ortalama Müşteri Sayısı", 
        min_value=0, 
        max_value=ort_musteri_sayisi * 3, 
        value=ort_musteri_sayisi
    )
    gelecek_toplam_miktar = st.slider(
        "Gelecek Aylık Ortalama Satış Adedi",
        min_value=0,
        max_value=ort_satis_adedi * 3,
        value=ort_satis_adedi
    )

    if st.button("Senaryoyu Simüle Et", type="primary", use_container_width=True):
        with st.spinner("Senaryo analizi yapılıyor..."):
            # 1. Baseline Tahmin (varsayımlar olmadan, sadece geçmiş ortalamalarla)
            _, baseline_tahmin = prophet_tahmin(aylik_satislar.copy(), senaryo_periyodu, gelecek_regresorler=None)

            # 2. Senaryo Tahmini (kullanıcı girdileriyle)
            senaryo_varsayimlari = {
                'musteri_sayisi': gelecek_musteri_sayisi,
                'toplam_miktar': gelecek_toplam_miktar
            }
            _, senaryo_tahmini = prophet_tahmin(aylik_satislar.copy(), senaryo_periyodu, gelecek_regresorler=senaryo_varsayimlari)

            # Sonuçları session state'e kaydet
            st.session_state.baseline_tahmin = baseline_tahmin
            st.session_state.senaryo_tahmini = senaryo_tahmini

    if 'senaryo_tahmini' in st.session_state:
        baseline_tahmin = st.session_state.baseline_tahmin
        senaryo_tahmini = st.session_state.senaryo_tahmini

        st.subheader("Senaryo Karşılaştırma Grafiği")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=baseline_tahmin['ds'], y=baseline_tahmin['yhat'], name='Mevcut Trend Tahmini (Baseline)', line=dict(color='grey', dash='dot')))
        fig.add_trace(go.Scatter(x=senaryo_tahmini['ds'], y=senaryo_tahmini['yhat'], name='Senaryo Tahmini', line=dict(color='green')))
        fig.update_layout(title="Mevcut Trend ve Senaryo Tahminlerinin Karşılaştırması", yaxis_title="Tahmini Ciro (€)")
        st.plotly_chart(fig, use_container_width=True)
        
        baseline_toplam_ciro = baseline_tahmin['yhat'].tail(senaryo_periyodu).sum()
        senaryo_toplam_ciro = senaryo_tahmini['yhat'].tail(senaryo_periyodu).sum()
        fark = senaryo_toplam_ciro - baseline_toplam_ciro
        
        st.subheader("Senaryonun Finansal Etkisi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mevcut Trend Toplam Ciro", f"{baseline_toplam_ciro:,.0f} €")
        col2.metric("Senaryo Toplam Ciro", f"{senaryo_toplam_ciro:,.0f} €")
        col3.metric(f"Senaryonun {senaryo_periyodu} Aylık Ciroya Etkisi", f"{fark:,.0f} €", delta_color="off")