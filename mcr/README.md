## Multiple-choice Retrieval

All of our mcr datasets are already pre-processed and are under this folder already along with git clone.

The only exception is the CSTS dataset, which is designed to not be publicly available.

Please attain access (not that diffucult) and download the CSTS original format from the [CSTS repo](https://github.com/princeton-nlp/c-sts), and run our code to create the two versions we design (`CSTS-hard` and `CSTS-easy`) for evaluation on RAR-b.

```bash
python create_sts.py

```

Notably, we use CSTS's validation set, just like other datasets where the test sets are not designed to be publicly available.