# Entrega dos resultados do E1

O pacote preserva os artefatos e a análise E1-A anteriores e acrescenta a análise completa do E1-B.

- `docs/scientific/e1/e1.md` — relatório E1-A + E1-B e conclusão do experimento.
- `docs/scientific/e1/e1_assets/` — figuras, CSVs e JSON agregados das duas análises.
- `docs/scientific/scripts/analisar_e1a.py` — script preservado para E1-A.
- `docs/scientific/scripts/analisar_e1b.py` — script reproduzível de consolidação E1-B.

Copie a árvore `docs/` do pacote para a raiz do repositório. Isso substitui somente os arquivos de relatório e análise que tenham os mesmos nomes. **Nenhum arquivo de treino da ADArena nem resultados do E0 são modificados**.

Para reproduzir os agregados do E1-B com as pastas originais de comparação:

```bash
uv run python docs/scientific/scripts/analisar_e1b.py \
  --input models/e1/comparacoes \
  --output docs/scientific/e1/e1_assets
```

Alternativamente, use `--input comparacoes.zip` (com a estrutura `comparacoes/seed-.../resultados.json` e `.csv`). Os intervalos bootstrap reamostram as seeds completas, sem tratar as quatro rodadas de cada seed como independentes.

**Limites:** resultados de validação offline contra o mesmo atacante corrente; não demonstram generalização para tráfego real, outros ataques ou outras distribuições.
