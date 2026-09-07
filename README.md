# Bybit Sinyal Botu - Kurulum

## Bot ne yapiyor?
- Bybit USDT-M perpetual piyasasinda listedeki coinleri (ve varsa hisse
  perpetuallarini) tarar.
- Strateji: 1G + 4S trend teyidi ile 1S RSI/MACD kesisimi + hacim patlamasi +
  ATR filtresini birlestirir.
- Her saat basi (GitHub Actions ile) calisir, yeni bir sinyal bulursa
  terambos44@gmail.com adresine mail atar.
- Ayni sinyali (ayni mum icin) tekrar tekrar mail atmaz.

## Kurulum Adimlari

### 1. GitHub reposu olustur
- github.com uzerinde yeni, **private** (gizli) bir repo olustur, orn: `bybit-sinyal-botu`
- Bu 4 dosyayi/klasoru repoya yukle:
  - `bybit_signal_bot.py`
  - `requirements.txt`
  - `state.json`
  - `.github/workflows/signal-bot.yml`

### 2. Secrets (gizli anahtarlar) ekle
Repo sayfasinda: **Settings > Secrets and variables > Actions > New repository secret**

Su 3 secret'i tek tek ekle:

| Secret adi           | Deger                                      |
|-----------------------|---------------------------------------------|
| `GMAIL_ADDRESS`        | Gonderen gmail adresin                      |
| `GMAIL_APP_PASSWORD`   | 16 haneli Gmail uygulama sifren             |
| `TO_EMAIL`             | terambos44@gmail.com                        |

**Onemli:** Sifreni asla dosyalarin icine yazma, sadece bu Secrets bolumune ekle.

### 3. Actions'i aktif et
- Repo sayfasinda **Actions** sekmesine gir, "I understand my workflows, go ahead and enable them" gibi bir onay cikarsa onayla.
- Workflow otomatik olarak her saat basi (XX:05) calismaya baslar.

### 4. Test et
- Actions sekmesinde "Bybit Sinyal Botu" workflow'unu bul, sag ustten
  **Run workflow** ile elle bir kere calistir.
- Calisma bitince loglara bak: hangi coinlerin tarandigini, hangilerinin
  bulunamadigini (ornegin bazi hisse sembolleri) orada gorebilirsin.
- Eger o an bir sinyal kosulu olusmadiysa mail gelmez, bu normaldir - bot
  calisiyor demektir.

## Coin/Sembol Listesini Degistirmek
`bybit_signal_bot.py` dosyasinin en ustunde `CRYPTO_SYMBOLS` ve
`STOCK_CANDIDATES` listeleri var, oradan coin ekleyip cikarabilirsin.

## Onemli Not
Bu bot **otomatik alim/satim yapmaz**, sadece mail ile bilgilendirir.
Islem karari ve risk yonetimi sana ait. Sinyaller yatirim tavsiyesi degildir.
