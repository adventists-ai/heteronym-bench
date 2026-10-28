# heteronym-bench：TTS 多音词读音测试集（英 / 日）

同一个写法在不同语境下读音不同（英文 *read* /riːd/ 与 /rɛd/，日文「今日」きょう与こんにち）。这个测试集检查语音合成在这类词上有没有按语境读对，而不是一律读成最常见的读音。

由 AI 降临派（adventists.ai）在开发语音对话模型时整理，用作内部读音回归测试。

[English summary below](#english-summary)

## 内容

| 语言 | 文件 | 题数 | 目标词 | 少数读音题 | 判读方式 |
|---|---|---|---|---|---|
| 英文 | `data/readset_en_v2.json` | 566 句 | 30 个同形异音词 | 255 | 音素 CTC（wav2vec2-lv-60-espeak） |
| 日文 | `data/readset_ja_v2.json` | 590 句 | 35 个多读音汉字词 | 340 | Whisper-large-v3-turbo 似然 |

- **少数读音**：正确读音不是该词默认读音的题。只会"照字面读"的系统在这部分会掉得最厉害，建议主要看这一项。
- 题目是纯文本，没有配录音。中文多音字集还在整理，见文末说明。
- `hold: true` 的题目对应的词，在我们自己的训练里被留出，可以用来测"没见过的词"。外部使用时可以忽略这个字段。

### 条目格式

```json
"ja_w08_11": {"lang": "ja", "text": "主人公は舞台の上手から静かに登場した。", "word": "上手", "pos": [7, 9],
  "answer": "かみて", "cands": ["うわて", "かみて", "じょうず"], "minority": true, "hold": false, "src": "v1"}
```

`pos` 是目标词在 `text` 里的字符区间 [start, end)。每个读音的定义在 `words[word].readings` 里：英文有 `ipa`（判读用）、`arpa`（CMU 音素）、`resp`（拼读写法）、`gloss`（词义）；日文有 `kana`（可接受的假名，可能不止一个）和 `gloss`。

## 用法

用你的 TTS 把每道题的 `text` 合成出来，按 `<条目 id>.wav` 存到一个文件夹（每种语言一个），然后运行：

```bash
pip install -r requirements.txt
python eval_enja.py --lang en --wav_dir out/en      # 默认判读器 ctcfuse
python eval_enja.py --lang ja --wav_dir out/ja      # 默认判读器 whisper
```

输出里的指标：
- `acc`：全部题目；
- `acc_minority` / `acc_majority`：少数读音题 / 默认读音题；
- `acc_reliable_words`：只算判读器可靠的词（见下）；
- `pred_default_rate`：判成默认读音的比例，接近 100% 说明系统基本只会读默认音。

逐题结果也会一并写出。

模型默认从 Hugging Face 下载，也可以用 `RM_WHISPER`、`RM_W2V` 指向本地目录。英文判读器另需 `phonemizer` 和 `espeakng-loader`。

## 判读器

判读器的输入是 16 kHz 音频、句子和候选读音，它给每个候选打分，取分最高的作为判定读音。

- **日文 `whisper`**：把句子里的目标词分别换成各候选的平假名（其余部分保留原文），用 Whisper teacher forcing 算整句对数似然。
- **英文 `ctcfuse`**：两项相加：
  - 整句的音素 CTC 似然，目标词的音素换成各候选的 IPA；
  - 把目标词前后各留 60 ms 切出来，单独再打一次分。切出来以后模型听不到上下文，不能"按语境猜"。

### 判读器验证（`validation/`）

用 TTS 把每道题的每个候选读音都强制合成一遍，正确读音和错误读音都有，再看判读器能不能还原出实际合成的读音。这一步是在 v1 的句子上做的：

| 判读器 | Kokoro（音素精确） | CosyVoice3 | 合计 |
|---|---|---|---|
| 英文 ctcfuse | 95.4%（读对 96.7 / 读错 94.2） | 89.6%（95.8 / 83.3） | 92.5% |
| 日文 whisper | — | 99.7%（299/300） | 99.7% |

- 英文有 6 个词的判读可靠度低于 0.85：invalid、bow、use、close、estimate、excuse。大多是 s/z 词尾清浊，或重音和弱读元音的差别。逐词可靠度见 `validation/judge_reliability.json`，看英文结果时建议同时看 `acc_reliable_words`。
- 日文在 v2 全部 590 题上做了一个反向检查：
  - 目标词写成**正确**读音的假名再合成，判读器全部判对（100%）；
  - 写成**错误**读音的假名（抽 118 题），判读器全部判为"读错"（0%）。

## 基线（同一判读器）

下表全部用本仓库的 `eval_enja.py` 跑出（2026-10-28）。合成时每种语言用同一个参考说话人。

| 系统 | 英文 全部 / 少数读音 / 可靠词 | 日文 全部 / 少数读音 |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base（零样本，x-vector 克隆） | 90.3 / 82.7 / 91.6 | 47.6 / 23.5 |
| 我们的语音对话说话头（32B，内部版本 P1v，经 E2 解码出声） | 88.2 / 80.0 / 89.9 | 77.8 / 68.5 |
| 判读器自检：Qwen3-TTS 读目标词改写成**正确**假名的句子 | — | 100 / 100 |
| 判读器自检：目标词改写成**错误**假名（抽 118 题） | — | 0 / 0 |

- 日文判成默认读音的比例（`pred_default_rate`）：Qwen3-TTS 78.1%，我们的系统 56.3%。只看文字的 TTS 大多读成默认音。
- 日文的错误主要出在"选哪个读音"，发音本身没问题：把目标词换成正确假名后可以全部读对。

## 数据来源与许可

- **英文**：
  - v1 的 120 句和 v2 新增的 179 句少数读音句由 Claude 撰写；
  - 267 句来自 Wikipedia Homograph Data（Google，句子出自 Wikipedia，CC BY-SA 3.0）。
- **日文**：
  - v1 的 144 句和 v2 新增的 206 句由 Claude 撰写；
  - 240 句来自青空文庫振り仮名语料（底本为公有领域作品）。
- **v1 复核**：v1 的句子用另一个 LLM（gpt-oss-20b）复核过。问的是目标词在句中是哪个词义，选项正反各排一次、问两遍；有分歧的逐条人工检查过。两次回答记在 `llm_check` 字段里。
- **许可**：代码采用 Apache-2.0。因为含 Wikipedia 来源的句子，数据整体采用 CC BY-SA 4.0（见 `LICENSE-DATA`）。

## 关于中文

我们内部还有一套中文多音字集，参考读音来自 MFA 对齐真人录音。发布前抽查发现对齐结果里有不少误标，例如部分句子里的"都""的""了"被标成 dū、dí、liǎo。为保证题目正确，这次没有放进来，人工校对之后再补。

## English summary

A heteronym reading benchmark for TTS in English (566 sentences, 30 heteronyms) and Japanese (590 sentences, 35 multi-reading kanji words). Each item gives the sentence, the target span, candidate readings and the context-correct answer. The main metric is accuracy on the **minority-reading** subset (answer ≠ the word's default reading).

Judges are automatic. They were validated on v1 sentences by forcing a TTS to say every candidate reading:
- EN: phoneme CTC with wav2vec2-lv-60-espeak, full sentence plus a context-free crop of the target word; 92.5% overall, 6 words below 0.85 reliability.
- JA: Whisper-large-v3-turbo likelihood with the target rewritten in each candidate kana; 99.7%.

Usage: `python eval_enja.py --lang {en,ja} --wav_dir DIR`, with files named `<item_id>.wav`. A Chinese polyphone set will be added after manual label review.

License: code Apache-2.0; data CC BY-SA 4.0. Sources: Wikipedia Homograph Data (CC BY-SA 3.0), the Aozora Bunko furigana corpus (public-domain texts), and sentences written by Claude.
