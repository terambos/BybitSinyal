Kripto Sinyal Botu - Kurulum
Bot ne yapiyor?
Fiyat verisini Yahoo Finance'tan ceker (Binance ve Bybit'in genel
API'leri GitHub Actions gibi bulut sunuculardan gelen istekleri
engelledigi icin veri kaynagi olarak Yahoo Finance kullanildi; buyuk
coinlerin fiyatlari borsalar arasi pratikte ayni hareket eder, sen
istedigin borsada - orn. Bybit'te - islem acabilirsin).
Strateji: 1G + 4S trend teyidi ile 1S RSI/MACD kesisimi + hacim
patlamasi + ATR filtresini birlestirir.
Her saat basi (GitHub Actions ile) calisir, yeni bir sinyal bulursa
terambos44@gmail.com adresine mail atar.
Ayni sinyali (ayni mum icin) tekrar tekrar mail atmaz.
Kurulum Adimlari
1. GitHub reposu
Zaten olusturdugun repo kullanilacak. Su dosyalarin hepsi guncel halleriyle
repoda olmali:
`bybit_signal_bot.py`
`requirements.txt`
`state.json`
`.github/workflows/signal-bot.yml`
Her dosyayi GitHub'da actigin sayfada kalem (duzenle) ikonuna basip,
Not Defteri'nde actigin guncel dosyanin icerigini yapistirip
"Commit changes" ile kaydet.
2. Secrets kontrolu
Settings > Secrets and variables > Actions altinda su 3 secret olmali
(PROXY_URL artik gerekli degil, silebilirsin, dursa da zarari yok):
Secret adi	Deger
`GMAIL_ADDRESS`	Gonderen gmail adresin
`GMAIL_APP_PASSWORD`	16 haneli Gmail uygulama sifren
`TO_EMAIL`	terambos44@gmail.com
3. Test et
Actions sekmesinde "Kripto Sinyal Botu" workflow'unu bul, Run workflow
ile elle calistir.
Yesil tik cikarsa bot calisiyor demektir. Kirmizi X cikarsa hata metnini
kontrol et.
Coin Listesini Degistirmek
`bybit_signal_bot.py` dosyasinin basindaki `CRYPTO_SYMBOLS` listesinden
coin ekleyip cikarabilirsin. Yahoo Finance formati onemli: coin kodu + "-USD"
(orn. "BTC-USD"), script bunu otomatik ekliyor, sen sadece "BTC" gibi
kisa kodu yaziyorsun.
Onemli Not
Bu bot otomatik alim/satim yapmaz, sadece mail ile bilgilendirir.
Islem karari ve risk yonetimi sana ait. Sinyaller yatirim tavsiyesi degildir.
