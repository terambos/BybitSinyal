name: Backtest

on:
  workflow_dispatch: {}

jobs:
  run-backtest:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Kodu indir
        uses: actions/checkout@v4

      - name: Python kur
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Bagimliliklari kur
        run: pip install -r requirements.txt

      - name: Backtest calistir
        run: python backtest.py

      - name: Sonuclari yukle (indirilebilir)
        uses: actions/upload-artifact@v4
        with:
          name: backtest-sonuclari
          path: results/backtest_sonuclari.csv
