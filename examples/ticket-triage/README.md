# Ticket triage

Five real Banking77 support tickets, three decisions in one Statim request: the intent (77 ways), how urgent it is, and whether the customer asks for money back. Uncertain answers come back with `escalate: true`.

The messages are the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) test split. This directory does not contain ticket text. `triage.py` downloads the parquet file on first use and caches it in `.cache/` (gitignored).

Python 3.10 or newer. The script uses the standard library and the client in [`clients/python`](../../clients/python). From a checkout it adds that package to `sys.path` itself. `pip install ./clients/python` works too.

## Ten minutes

From a new directory:

### 1. Get Statim

Release binary (Linux x86-64, CPU):

```sh
git clone --recursive https://github.com/BEKO2210/statim.git
cd statim
curl -fL -o statim-0.9.1-linux-x86_64-cpu.tar.gz \
  https://github.com/BEKO2210/statim/releases/download/v0.9.1/statim-0.9.1-linux-x86_64-cpu.tar.gz
tar -xzf statim-0.9.1-linux-x86_64-cpu.tar.gz
```

The binary is `statim-0.9.1-linux-x86_64-cpu/statim`. Checksums are attached to the [v0.9.1 release](https://github.com/BEKO2210/statim/releases/tag/v0.9.1).

Build from source instead:

```sh
git clone --recursive https://github.com/BEKO2210/statim.git
cd statim
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The binary is `./build/statim`.

### 2. Get the model

The multilingual decision model is [`Beko2210/statim-decide-multilingual-base`](https://huggingface.co/Beko2210/statim-decide-multilingual-base). The CPU file is `statim-decide-multilingual-base-q8_0.gguf`:

```sh
curl -fL -o statim-decide-multilingual-base-q8_0.gguf \
  https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/statim-decide-multilingual-base-q8_0.gguf
```

`SHA256SUMS` in that repo is the checksum list. A GGUF you already have is the same kind of argument. Point `-m` at the file:

```sh
statim serve --device cpu -m multilingual=/absolute/path/to/model.gguf --port 8080
```

### 3. Start the server

```sh
./statim-0.9.1-linux-x86_64-cpu/statim serve --device cpu \
  -m multilingual=statim-decide-multilingual-base-q8_0.gguf \
  --port 8080
```

Use `./build/statim` if you compiled it. Wait until `curl -s localhost:8080/health` returns `{"status":"ok",...}`.

### 4. Demo

```sh
cd examples/ticket-triage
python3 triage.py demo
```

`--url` defaults to `http://127.0.0.1:8080`, `--model` to `multilingual`, `--seed` to `0`, `--min-confidence` to `0.8`. Each ticket is one `POST /v1/systemone` with three questions.

### 5. Eval

```sh
python3 triage.py eval --limit 500
```

`--limit` is a seeded stratified sample: every intent appears, and the counts differ by at most one. Omit `--limit` to run all 3,080 test rows. `--concurrency` defaults to 4 (a thread pool of separate requests). The command writes `results.json` and `results.md` next to `triage.py`.

Intent names are sent as readable phrases (`card_arrival` becomes `card arrival`), which is the label text this model was trained on. The state is `{"message": "<ticket>"}`. The script sends `head_max_len: 512` because 77 options do not fit a 256-token option budget: below that, the server keeps only a few subwords of each option.

## What you should see

Measured on an AMD Ryzen 7 5800X (8 cores / 16 threads), 16 GB RAM, Linux, CPU only, with the
published 0.7.0 `statim-decide-multilingual-base-q8_0.gguf` (SHA-256
`96fb3971656ee6b596cb0c108aff4bbe1306d87707ffe9ae5f364a52b919ff1b`, as listed in the model's
`SHA256SUMS`): the evaluation on 2026-09-28 with Statim 0.7.0 and one inference worker with 16
threads, the `demo` transcript on 2026-09-29 with Statim 0.8.1.

`demo` (`--seed 0`, `--min-confidence 0.8`). All five intents match the gold label:

```text
[1/5]
text     Please delete my account.
intent   terminate account   p=1.0000   gold=terminate account (terminate_account)   match   escalate=no
urgency  immediately   score=1.6762   p=0.7018   escalate=yes
refund   no   noul=0.0327   escalate=no
ticket   escalate=no (intent at or above 0.80)

[2/5]
text     There's a debit on my account that I didn't do.
intent   direct debit payment not recognised   p=1.0000   gold=direct debit payment not recognised (direct_debit_payment_not_recognised)   match   escalate=no
urgency  today   score=0.9616   p=0.8329   escalate=no
refund   no   noul=0.0751   escalate=no
ticket   escalate=no (intent at or above 0.80)

[3/5]
text     Where do I order a virtual card from?
intent   getting virtual card   p=1.0000   gold=getting virtual card (getting_virtual_card)   match   escalate=no
urgency  today   score=1.0198   p=0.6155   escalate=yes
refund   no   noul=0.3285   escalate=yes
ticket   escalate=no (intent at or above 0.80)

[4/5]
text     Where is my card accepted?
intent   card acceptance   p=1.0000   gold=card acceptance (card_acceptance)   match   escalate=no
urgency  today   score=0.6162   p=0.4677   escalate=yes
refund   no   noul=0.2909   escalate=yes
ticket   escalate=no (intent at or above 0.80)

[5/5]
text     If I want a physical card, do I have to pay anything?
intent   order physical card   p=1.0000   gold=order physical card (order_physical_card)   match   escalate=no
urgency  can wait   score=0.6683   p=0.5208   escalate=yes
refund   no   noul=0.1946   escalate=no
ticket   escalate=no (intent at or above 0.80)
```

Only the routing decision, the intent, decides whether a ticket goes to a person: a ticket is
escalated when the intent answer has `escalate: true` (the server's rule,
`answer_confidence < min_confidence`). Urgency is an ordinal score whose probability naturally
spreads over neighbouring levels, and the refund flag is a side signal, so their own flags are
shown but do not escalate the ticket.

`eval --limit 500 --concurrency 2` then asks the intent question only (seed 0, 77 intents, six or
seven tickets each, warm process):

Intent accuracy **0.9080** (454/500).

| min_confidence | coverage | answered | accuracy | correct |
|---:|---:|---:|---:|---:|
| 0.0 | 1.0000 | 500 | 0.9080 | 454 |
| 0.5 | 0.9880 | 494 | 0.9150 | 452 |
| 0.6 | 0.9880 | 494 | 0.9150 | 452 |
| 0.7 | 0.9880 | 494 | 0.9150 | 452 |
| 0.8 | 0.9840 | 492 | 0.9167 | 451 |
| 0.9 | 0.9800 | 490 | 0.9204 | 451 |

Coverage is the share whose `answer_confidence` is at least the threshold (the share the server
would not escalate); accuracy is computed on that share. A threshold of 0 escalates nothing.

End-to-end latency at concurrency 2, including queueing behind the single worker: **p50 439.6 ms**,
**p95 488.3 ms**; throughput **4.49 requests/s** (500 requests in 111.2 s). With one worker, two
in-flight requests each wait about one service time. A GPU build answers far faster; see the main
README. The first request after startup is slower while the weights are paged in.

The model is usually very sure: 490 of 500 intents have `answer_confidence` of at least 0.9, so a
threshold of 0.9 sends 10 tickets to a person and lifts accuracy on the rest from 0.908 to 0.920.
The full 3,080-row test split or the f32 file print slightly different numbers. `results.json` and
`results.md` in this directory are this run.

## What this is

Statim reads the customer message once and returns a typed intent, a three-level urgency score (`can wait`, `today`, `immediately`), and a yes/no answer to “Does the customer ask for money back?”. The intent choices are the 77 Banking77 labels. Selective prediction is the `escalate` bit on each answer, and the eval curve is that rule applied to the returned `answer_confidence`.

Banking77 labels intent only. Urgency and refund in the demo are extra questions with no gold label here, so the example does not score them. On the five demo tickets the urgency score was under 0.8 every time, and the ticket was escalated even though the intent was certain. The curve is one seeded sample of 500, not a calibration guarantee and not a rank-based “keep the most confident 60%” cut. The server in this measurement had no API key and served a single model on CPU.

## Data

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) test split, 3,080 messages, 77 intents, 40 each. Licence CC-BY-4.0. The loader reads the Hugging Face parquet (`data/test-00000-of-00001.parquet` on `main`, then the same file on the open parquet conversion `refs/pr/7` until that lands). It checks the file for 3,080 rows and 77 class names before using it.

Attribution: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić, “Efficient Intent Detection with Dual Sentence Encoders”, *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI*, ACL 2020. <https://arxiv.org/abs/2003.04807>. Data also at <https://github.com/PolyAI-LDN/task-specific-datasets>.
