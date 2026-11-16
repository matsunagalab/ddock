# 予測精度改善：探索・選別・変形の切り分け

2026-11-16、調査時HEAD `de42131`。これは担当Cの調査・評価診断メモである。
本番予測の成功を新たに得たという報告ではない。研究履歴の正本はrootが追記する
`EXPERIMENT_REPORT.md` §9.21。重い全探索、訓練、nativeを用いた係数調整は行っていない。

## 結論

最優先は、**接触・内部strainの物理モデルを検証しながら、候補が失われる段階を観測可能にすること**。
128回転の粗さは実在するが、「回転数が少ないからacceptableは出せない」という説明だけでは不十分だった。
独立評価で、既存の128回転／2 Å並進格子にDockQ 0.3677のapo剛体配置を構成できた。
その配置はfull-support条件を満たすが現スコアでは非常に不利で、各回転のTop4選択ですでに落ちる。
ただしこのoracle配置にもclashがあり、スコアが高いことだけを誤りと断定して反発を弱めてはいけない。

同時に、両apoをnativeへ独立剛体fitするだけでDockQ 0.9101になる。
この系を修正開発に使うのはよいが、誘導適合の有効性は別のmedium/difficult apo集合で検証する必要がある。

## 確認した既存実装・結果

- `src/zdock/inducedfit/search.py`: 各回転で3 Å以内の並進NMS後4 peaksを選び、全回転をenergy順にまとめてTopKへ切る。回転間の構造clusteringはない。現runnerは残った候補しかPDB保存しない。
- pilot v1はidentity＋seed0の127回転、Top5、receptor 2 modes、最大局所回転0.15 rad、最大局所並進0.5 Å。全15精密化は8 iteration到達で、収束済みではない。
- R1 best5=0.201863、R2 best5=0.157074。R2はR1の相対的に良い候補を失った。最終energy順位が同じ誤Top1を選ぶ。
- v2の実変形検証は近似応答の過大評価を抑えたが、Top1/best5を改善しなかった。これは近似の整合性修正と構造精度を区別すべき例。
- `scripts/eval_search_ceiling.py` は旧スコア／holo cache用。そのnearest quaternion＋固定のsnapped translationを**厳密なDockQ上界と呼ぶことはできない**。角度最小とDockQ最大は同一でなく、最適並進は回転・分子形状にも依存する。新apo診断にはそのまま流用しない。

根拠は `docs/apo_pilot_v1.md`、`docs/apo_pilot_v2.md`、両results CSV、研究ログ§9.18–9.20、および上記コードの直接読取。

## 今回実施した小規模な独立評価

### 問い・条件・漏洩の扱い

既存128回転がapoのnative相対配置をどれくらい粗く量子化しているかを、凍結済みrunの後に測った。
入力は1A2P B／1A19 A、nativeは1BRS A/D。公式DockQと同じ配列対応を使い、配列一致したCAで
各apoをnative monomerへ独立Kabsch fitした（R108 CA、L86 CA）。その2 fitを合成し、元runnerの
独立重原子centeringを保った相対姿勢を作った。nativeの内部形状をapoへ移植していない。

各保存quaternionについて、(a)上記apo fitとの回転角、(b)その回転でligand CAの最小二乗並進、
(c)その並進を2 Å格子へ丸めた残差を測る。最近接角と最小CA残差のprobeはこの系では同じindex41だった。
これらは**nativeを使った事後診断**であり、本番予測やDockQの探索上界ではない。
生成PDBとnativeは `data/inducedfit/evaluator_only/` のみに保存。本番manifest、回転、mode、重みは変えない。
nativeから変形モデルや乱数seedを選ばず、学習loss・checkpoint選択もない。

実装：`scripts/diagnose_inducedfit_pose_coverage.py`。
保存先：`data/inducedfit/evaluator_only/pose_coverage_v1/`。
macOS27.0.1 arm64、Python3.12.11、NumPy1.26.4、既存のDockQ2.1.3隔離環境。
DockQは各child `--n_cpu 1`、60秒上限、固定 `RL:AD`／allowed mismatches1。
全体1.02884秒。native sanity＋4 probesがすべて採点完了、失敗／skip／timeoutは0。

| 指標 | 実測 |
| --- | ---: |
| 受容体の独立fit後全対応CA RMSD | 0.485697 Å |
| ligandの独立fit後全対応CA RMSD | 0.438757 Å |
| 保存回転の最近接角 | 30.5154895°（index41） |
| 現局所回転上限 | 8.594367°（0.15 rad） |
| index41の最小二乗並進 | [9.797334, 1.001535, 21.359511] Å |
| その2 Å格子丸め | [10, 2, 22] Å |
| 並進丸め誤差 | 1.203425 Å |
| index41の対応ligand CA RMSD（最適並進／格子並進） | 5.201585／5.338981 Å |
| 正確apo fitの相対並進 | [9.735346, 0.848536, 21.382761] Å |
| 正確apo fitの並進丸め誤差 | 1.082145 Å |

CA残差はこの独立fitでの値であり、下表の公式DockQ LRMSDとは定義が異なる。

| 評価専用oracle配置 | DockQ | iRMSD Å | LRMSD Å | fnat | DockQ clashes |
| --- | ---: | ---: | ---: | ---: | ---: |
| apo独立fit、連続並進 | 0.910119 | 0.507134 | 0.498443 | 0.836364 | 5 |
| apo独立fit、格子丸め並進 | 0.849820 | 0.696078 | 1.176763 | 0.745455 | 3 |
| 最近接角index41、格子丸め並進 | 0.367706 | 3.133156 | 5.344706 | 0.200000 | 3 |
| 128回転中の最小CA残差、格子丸め並進 | 0.367706 | 3.133156 | 5.344706 | 0.200000 | 3 |

native対nativeのDockQ=0.999999999999687。最後の2行は独立した成功例ではなく同じ配置。
最近接角probeはnative contact11/55、nonnative contact31/42である。DockQ acceptableは
衝突がないことも正しい化学も保証しない。剛体oracle自体のclashはapo側鎖差・丸め・構造差等を含む。
30.5°の向き誤差を現在の0.15 rad局所更新1回では解消できないが、回転誤差だけを全失敗の原因とも言えない。

### 同じ全H・場・gridでの選別位置診断

rootへの事前連絡後、index41だけを元の全H再構築／場／83³格子で再計算した。
本番と同じ `valid_translation_mask` と `search_response(selection='rigid')` を使い、
4 peaks／3 Å NMSを維持。nativeは読み直さず、上のeval-only記録のprobeだけを入力とするが、
情報の由来はnative-informedである。全128回転の再探索はしない。

`data/inducedfit/evaluator_only/pose_coverage_score_v1/` に保存、6.52141秒で完了。
Python3.12.11／Torch2.13.0、CPU float64、PyTorch1thread。BLAS thread数は独立未確認。
full-support=True、FFT energy=+1150.368826661 ε、直接energyとの差1.82e−12 ε。
ここで直接energyはligand座標を直接shiftして同じ場を採点した値で、静電potentialの畳み込みには
FFTを使う。独立した全原子pair energyとの一致を測ったという意味ではない。
valid並進188,384点のうち175,900点がprobeより低energyだった。

| index41の本番Top4並進 | rigid energy ε |
| --- | ---: |
| [−14, −22, 10] | −65.124452 |
| [16, 4, −26] | −62.987892 |
| [−14, −20, 14] | −62.450299 |
| [−10, −26, 2] | −58.884709 |

probeは格子外ではなく、**回転内Top4選別ですでに落ちる**。
元R0のglobal Top5 cutoffは−84.101241 εなので、この回転のTop4自体も最終Top5より不利。
どの段階にもscoreの選別が関わるが、この1 probeを救うために重みを合わせることはしない。
次は真の接触距離を保つ小分子／平面試験と、正解近傍の力の向きを別に検証する。

### 再現と検証

```bash
data/inducedfit/dockq_env/bin/python scripts/diagnose_inducedfit_pose_coverage.py \
  --run-report data/inducedfit/apo_pilot_v1/pilot_run/report.json \
  --native data/inducedfit/evaluator_only/1BRS.pdb --native-chains AD \
  --allowed-mismatches 1 --dockq data/inducedfit/dockq_env/bin/DockQ \
  --out data/inducedfit/evaluator_only/pose_coverage_v1
.venv/bin/python scripts/diagnose_inducedfit_pose_coverage.py \
  --run-report data/inducedfit/apo_pilot_v1/pilot_run/report.json \
  --score-only-coverage data/inducedfit/evaluator_only/pose_coverage_v1/report.json \
  --out data/inducedfit/evaluator_only/pose_coverage_score_v1
```

生成先は新規directoryを要する。geometry run後にscore-only経路を同scriptへ追加したため、
両reportのscript SHAは異なる。元geometry計算式は不変、当時のhashは各reportへ保存した。
seed23の独立NumPy乱数16個でquaternion変換を本番 `zdock.geom.rotate` と照合し、
既知rigid fit回復16個とcollinear拒否も確認、全通過（絶対誤差閾値1e−12）。
これは軽量の数値checkで、全test suiteの再実行ではない。
2回の実測後、同じframe合成式をhelperへ分離し、非unit／nonfinite quaternionの明示拒否を追加した。
測定済みの正常入力に対する式は変えていない。nativeを使わない回帰testを
`tests/test_inducedfit_pose_coverage.py` に追加し、rootの最終suiteへ引き渡した。

主要SHA256：

- manifest: `969c76da6acb3e911b11e2d623b858c3096893f628f1012c71328dc47a36ff04`
- frozen run report: `e7d1c14e55e8b2c992a62b6df3489b9934dfc17467414f322ce68f39453dd16e`
- native: `0076aae54252706a8c8c02f7b1e71ce2394b0e928b413590b4ac12aade9e8f6c`
- apo R: `20e4bdaadc1ab001ba1b2bc55a17408412b9edc6e3f51847e40deb6f9f3bf02b`
- apo L: `2580ba081a5f257fb6b0605036489c8de050d972ba41c24ce58cbe3cfd30d316`

## 関連repo／一次資料から取り入れること

| 資料 | 確認内容 | ddockでの使い方・注意 |
| --- | --- | --- |
| [HADDOCK3 examples](https://github.com/haddocking/haddock3/blob/main/docs/pages/examples.md)、[clustfcc code](https://github.com/haddocking/haddock3/blob/main/src/haddock/modules/analysis/clustfcc/clustfcc.py)、[公式解析manual](https://www.bonvinlab.org/haddock3-user-manual/modules/analysis.html) | rigidbody→FCC clustering→clusterごとの候補選択→flexref/emrefという構成がある。predicted pose自身からcontactを作れる。 | global TopKの前にbasinの多様性を残す。native interfaceを使うclusteringと混同しない。HADDOCK3本体は[Apache-2.0](https://github.com/haddocking/haddock3/blob/main/LICENSE)。外部CNS等の条件は別途確認。 |
| [LightDock](https://github.com/lightdock/lightdock/blob/master/README.md)、[公式basics](https://lightdock.org/tutorials/basics)、[原著](https://doi.org/10.1093/bioinformatics/btx555) | 表面swarmで探索を分け、両partnerのANMを使う選択肢がある。通常のANMは各10非自明方向。 | 対照として有用。ddockの独自性はANM自体ではなく、相手力への応答をFFT全並進選別前へ入れること。GPLv3なのでMIT repoへコードを無条件コピーしない。 |
| [DB5.5公式](https://zlab.wenglab.org/benchmark/index.html) | rigid162／medium60／difficult35の計257 cases、両入力PDB・multimer・HETATM・難度情報がある。CC BY4.0。 | 少数成功例の拾い集めでなく、化学適格性監査と難度層別に使う。nativeに由来する難度分類は評価層別専用。 |
| [抗体benchmark repo](https://github.com/piercelab/antibody_benchmark/blob/master/README.md)、[原著](https://pmc.ncbi.nlm.nih.gov/articles/PMC8178184/)、[公式table](https://piercelab.ibbr.umd.edu/antibody_benchmark/) | 67件の抗体–抗原bound/unbound、CC BY4.0。READMEはbound/unboundを事前重ね合わせ済みと明記。 | **独立centeringだけでは不十分**。両apoへ独立seed固定SO(3)回転を施し、初期orientationの答え漏洩を防ぐ。 |
| [DockQ公式](https://github.com/wallnerlab/DockQ/blob/master/README.md) | mapping指定、matching-residue評価、JSON出力、clashes等を提供。 | DockQと衝突・化学・失敗分母を併記する。pose selectionへDockQを渡さない。 |

上記repo/公式資料を重点的に確認した調査であり、世界中のrepoの網羅的レビューではない。
旧UMass benchmark URLはweb取得不能だったため、同じWeng labの現公式mirrorを確認した。
抗体repo READMEのmain branch取得は失敗、masterを確認した。科学計算の失敗とは別の資料取得上の事項である。

## 優先する改善案と最小2段階実験（以下は未実施の提案）

### 段階1：候補がどこで消えるかを見えるようにし、scoreを物理的に直す

まずsearchの出力契約に、全回転Top4の軽量metadata保存を加える。既存128回転なら最大512候補で、
全格子や全候補PDBを保持する必要はない。各段階の集合を保存し、独立scorerで
`raw512 → selected50 → refined50 → final10` のbest DockQと成功候補保持率を測る。
「raw512のbest」は回転格子全体の上界ではなく、すでにscoreで切られた集合のoracle値である。
全128回転分を予測後の凍結manifestで評価し、native oracleを混入しない。

同じ予算の3 armだけを事前固定する：

1. 現score＋現在のenergy選択（対照）。
2. 単純系の接触距離・反発・力・格子収束を合格した新物理score＋同じ選択。
3. 2と同じscoreで、多様性を残す候補選択。predicted contact mapまたはligand相対配置でclusterし、
   各basin最低1候補を確保。rigidで有利な候補と応答で有利な候補に固定quotaを置き、
   R2がR1の全候補を捨てることを避ける。例として総pool50の25 rigid＋25 responseを先に固定し、
   重複除去後の不足補充規則も固定する。nativeによるquota調整はしない。

ここではangular set、mode数、局所budgetを変えない。最初にscoreを変えた効果を識別する。
停止条件は、物理単純系が不合格ならproteinの重み調整へ進まない、全原子strain／clashが増えて
DockQだけ上がるなら採用しない、raw集合にacceptableがなければ順位付けの成功と呼ばないこと。
機構が通った後に、512/2048等のinput-independentな階層SO(3)探索を別ablationとする。
多段探索はscoreの悪いbasinへ計算を集中させる恐れもあるため、探索の増加だけを先行させない。

### 段階2：小さい独立apoパネルで誘導適合を評価する

DB5.5全体を親母集団とし、まず入力側metadataだけで両partnerの実apo性・chain・欠損・
disulfide／金属／cofactor・残基数・force-field適格性を監査する。適格／不適格の全行を残す。
「用意できた12例」ではなく、この適用範囲を固定してから配列family単位で分割する。
例として開発6系、未使用評価12系（rigid／medium／difficult各4）をmetadata＋ID hashで固定。
1BRSおよび近縁familyは開発側へ隔離し、結果を見て失敗sampleを差し替えない。
12系では統計的なHDOCK優位性確定には小さすぎるが、改善方向の棄却・継続には使える。

推論用manifestにはapoだけを入れ、独立SO(3) scrambleのseedを予測前に固定する。
同じ対象・同じscrambleを全armへ渡す。native情報は独立evaluator、難度層別、事前の配列対応監査だけ。
R0 rigid、R1同じ候補の柔軟refine、R2v応答候補＋同じ柔軟refineを比較し、追加計算費用を明示する。
Top1/5/10成功率、raw-pool成功率、DockQ変化、clash、原子力場strain、全工程time/RSSを保存。
全登録系を分母としたend-to-end成功率と、成功preparationに限定した条件付き精度を両方示す。
準備不能・OOM・timeout・scorer失敗は別列で保持し、未採点を誤った0点と混同しない。

HDOCKは同じapo情報だけ、template利用の有無を区別し、既定実行と計算資源を合わせた実行を
別に報告する。異なる方法の「同じ回転数」を公平予算と呼ばず、同一hardware/thread・wall上限・
出力TopK・既知情報条件を比較し、時間–成功率curveを示す。LightDock ANMかHADDOCKの
非native restraint条件も物理flexible対照候補とする。外部手法が成功した候補をddockへ混ぜた実験は
再採点／精密化として明示し、ddock単独探索の成功率とは分ける。

## 抗体loopへ進む前の必要条件

現preparationはdisulfide／crosslink／複数chainに制限があり、通常抗体をそのまま評価できない。
disulfide topologyと力場、heavy/light鎖の相対配置保持、chain break／欠損loopの方針、
両端固定loop closure、側鎖再配置を先に実装・検証する。準備に通る抗体だけ事後選別しない。
最初のloop実験では抗体framework／抗原の剛体poseが既知という局所問題と、盲目的な全体dockingを
別の課題として報告する。CDRに基づく事前情報を使うなら対照にも同じ情報を与える。

結論として、今回の診断は「scoreが選べる正解があるか」「変形がそもそも必要か」を分ける具体的根拠になった。
この1系でnative oracleを回収する調整より、接触物理を直したうえで集合の脱落を記録し、
独立したmedium/difficult apoで応答の利点を確認する順序を推奨する。
