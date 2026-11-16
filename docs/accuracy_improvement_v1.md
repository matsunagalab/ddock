# ddock：予測精度を改善するための調査結果と実装順序

2026-11-16。対象HEAD `de42131`。3サブエージェントによる独立調査を統合。
研究履歴の正本は [EXPERIMENT_REPORT.md §9.21](../EXPERIMENT_REPORT.md)。
今回は改善案の調査と既存失敗の診断であり、新scoreの予測精度はまだ測っていない。

**最初に直すのは接触scoreと分子内変形コスト。その後、相手に応じて動ける方向を増やす。**
既存ANMの方向を単純に増やしたり、回転数だけを増やしたりする前に、
正しい距離へ接近する力と、無理な内部変形を防ぐ力を整える。

## 1. 今回得られた根拠

| 確認事項 | 観測 | 言えること／限界 |
| --- | --- | --- |
| 接触距離（§9.19の既存測定） | 現shapeのC–O極小6.1204 Å、幾何半径和3.22 Å。cross項だけでも5.1735 Å | 長いGaussian tail／shell形状に由来する問題。LJ案の基準は幾何半径和でなく、固定σ・εによる解析的極小にする |
| 内部strainの成分再読 | 問題のR2 3候補でvdW +50.9〜120.8、Coulomb +13.1〜54.0、bonded −4.12〜−3.55 kJ/mol | torsion追加だけでは説明不足。全30構造中の既知3失敗の事後診断であり、一般化した統計ではない |
| 2→8 ANM方向 | 最大原子変位の作用素norm 3.971→19.755。同じ原子上限1.5 Åなら共通振幅球0.3777→0.07593 Å | 一つの大きな変位方向が、全方向を約1/5へ絞る。モード増加が柔軟性増加になるとは限らない |
| 保存128回転の到達性診断 | q41と評価専用並進[10,2,22] ÅでDockQ=0.367706。元grid内でfull supportを満たす | 元の離散探索領域にacceptable判定の姿勢が少なくとも一つある。nativeから並進を作った診断で、予測成功ではない |
| その姿勢の現score | +1150.369 ε。q41 Top4は−65.12〜−58.88 ε、回転内で除外 | DockQ出力clashes=3、非native接触率約0.738。高scoreをすべて誤りとは断定できず、repulsionを弱める根拠にはしない |

独立apo fitを使う評価専用poseはDockQ0.910119、並進だけ2 Å格子へ丸めると0.849820。
現在の開発1系は、剛体fitが良い姿勢を表現できる系であり、難しい誘導適合の有効性を示す試験ではない。
詳細・再現command・全probeは[探索／評価調査](research/accuracy_sampling_evaluation.md)に記録。

## 2. 改善案と優先順位

### 案1：原子型の物理potentialをFFTの場にする

**狙い：離れた配置を好む接触モデルを修正する。第一実装候補。**

現在の同じOpenMM systemからchargeに加えてLJのσ・well depthを保存し、
probe原子型tごとに `R_t(x)=Σ_i V_it(|x-r_i|)` を作る。
ligand型の格子と相関をとり、同じRのJVPを変形応答へ渡す。
使用中ff14SB XMLには非零LJ parameter組が最大13種類あり、数百channelは不要。
ただし13 LJ＋電荷は今の3channelより高価で、特に微分FFT cacheが増える。

接触井戸を保つため、元LJの零点・極小を含む領域は変えず、深い衝突だけをC2に有限化する案を比較する。
現在の2 Å格子を流用できるとは仮定しない。pair potentialと格子の力・極小・位相収束を先に確かめる。
十分な解像度が全protein格子で高価なら、粗いFFTを候補提案に使い、
**同じ元potentialによる直接計算で検証・精密化**する。候補取りこぼしは別に測る。

LJを足しただけで溶媒中の結合自由エネルギーにはならない。電荷の埋没・脱溶媒和は別に必要。
現effective ε単位とkJ/molを混在させず、CA priorを含む全項のunitを定義する。
core–shell係数をこの1系のDockQに合わせて調整する方法は採らない。

根拠・式・13型監査・C2案・91型対の検証計画は[接触物理調査](research/accuracy_contact_physics.md)。
型別grid自体は[ATTRACT実装](https://github.com/sjdv1982/attract/blob/a45f6e086a42dc3f45e28e0c83a361466d79ba96/bin/grid_calculate.cpp)
などに先例があり、その新規発明とは主張しない。

### 案2：分子内nonbondedを含む変形コストに置き換える

**狙い：相互作用energyの得より大きな内部損傷を許さない。案1と並行して準備する。**

CAばねだけでは見えない分子内vdW・Coulomb・torsion・溶媒成分を扱う。
基準状態を曖昧にした `U(x(a))−U(x_apo)` の追加は避ける。
実apoは力場の最小点とは限らず、一次力や負strainが残るからである。

全pose・全比較armで共通の、apoのみを使う制限付き準備を定義する。
入力構造へのanchorを明示した局所基準点を求め、その近傍の物理energy差とKを使う案を第一候補とする。
これは単量体ensembleによる探索ではなく、誘導適合前の共通基準の準備である。
準備で重原子を動かした量、残差torque、Kの正定値性を記録し、成立しない系は失敗に数える。
CAばねへの無条件加算や負energyの0へのclipで整合したことにしない。

このprior自体をFFT化する必要はない。固定Kを小次元応答solverへ渡し、
実際に変形した候補では同じ全原子energyを直接検査できる。
1–2/1–3除外、1–4 scaling、protonationを元OpenMM systemと一致させる。

成分別表、基準点・単位・force接続の注意は[内部応答調査](research/accuracy_internal_response.md)。

### 案3：方向ごとの可動性を残す原子変位制約

**狙い：モードを増やしても全体が動けなくなる現象を避ける。案1・2の後。**

現在は最も大きく動く原子から、全方向共通の小さな振幅球を決める。
候補ごとに以下の小次元制約を解けば、安全な方向の余地を残せる。

```text
minimize    gᵀa + 1/2 aᵀKa
subject to  ||a|| <= requested_CA_radius
            ||B_i a|| <= max_atom_displacement   for every atom i
```

KがSPDなら凸問題。全FFT並進で解くのでなく、rigid候補と応答候補の固定quotaから作ったpoolに限る。
非線形座標へ戻した後のtrue CA変位・全原子最大変位・物理energy検証は維持する。
二次gainだけが増えて実strainも増えるなら不採用とする。

最初は凍結pool上で球制約との比較を行い、solver誤差・active原子・実gainを測る。
同じ可行集合に対する高速解法の既知の考え方を用いる案で、制約最適化そのものの新規性は主張しない。

### 案4：接触力が要求する局所運動を追加して、FFTへ戻す

**狙い：ddockの独自性を伸ばし、抗体loopへつなげる。前3案の後の研究候補。**

第一passの多様な候補に対し、既存ANMで吸収できない接触力を局所φ/ψ/χへ写し、
物理HessianのHVPと反復solveで少数の応答方向を作る。基底・K・微分場を再構築して第二passのFFTへ戻す。
第二pass中は基底を固定し、構築費用も比較予算に含める。

固定8 ANM、4 ANM＋4接触応答、4 ANM＋4 random局所方向を、同じ方向数・候補数・評価回数で比較する。
誤接触へ過適応しないようrigid/global候補を残す。
ANMも局所柔軟精密化も既存技術であり、[iATTRACT](https://pubmed.ncbi.nlm.nih.gov/25402278/)に先例がある。
研究上の差は、**相手が要求する変形とその物理コストを、FFTの候補選択へ返す**点に置く。
この組合せの新規性・有効性は今後検証する。

抗体ではloop両端の位置・向きとdisulfideを保つ必要がある。
constraint Jacobianのnullspaceだけで有限変位のclosureは保証できないため、
finite-step closure/retractionとχ回転を別途実装する。
現prepareはcrosslink拒否であり、そのまま抗体へ対応済みとは扱わない。

## 3. 実装・評価を進める順序

下記は次実験の提案。protein本run前に値・入力hash・budgetを登録して凍結する。

1. **物理と微分の検証**：案1の非零91 LJ型対＋zero-LJ型を全数検査。
   h=1/0.5/0.25 Åと各軸の0/half-cell phase、連続LJ対grid、FFT対同grid直接計算、JVP対差分を比較する。
   この値は小fixtureの検査条件で、proteinに0.25 Åを使う宣言ではない。
   同時に案2のapo残差torque／分子内pairコストを監査する。係数をnativeに合わせない。
2. **既知開発系の機構試験**：まずrigidのまま旧／新接触scoreを比較し、その後同じ新scoreでrigid／flexを比較。
   既存128回転・seed0を基準に、全候補poolを保存して、回転内peak除外・pool recall・最終順位を分離する。
   評価専用のnative-near配置を使う場合はoracle試験と明記し、blind予測成功率に混ぜない。
   例としてapo-fit poseと±10°の各軸回転、±2 Åの各軸並進からなる13開始poseを固定し、
   正解への接近、内部strain、接触回復を同じ評価予算で測る。これはまだ未実行。
3. **未使用apoパネル**：単一開発系から独立したeasy／moderate／difficultを、preparation前に固定する。
   例として各4件の12登録系から始める。適格条件・順序・family重複を先に定め、
   失敗系を都合のよい別系へ置換しない。準備失敗も全登録系の成功率の分母へ含める。
   blind推論を固定してから、別processで公式DockQ・interface幾何・clash・strainを評価する。
4. **比較と判断**：同回転／候補数の比較と同wall-timeの比較を区別し、Top1/5/10とpool recallを分ける。
   改善の有無は独立パネルで判断し、runtime・準備成功率・全失敗も報告する。
   HDOCKのtemplate-enabled／template-disabledを明記し、実行環境・version・入力を固定してから比較する。

**停止条件**：連続potentialの接触・力が不適切、grid/JVPが合わない、単位／基準点が不整合なら
大きな予測実験へ進まない。oracle配置だけ改善した場合もblind成功としない。
独立パネルの結果を見て係数を変えたら、そのパネルは以後developmentであり新testを用意する。

## 4. 漏洩と研究主張

[抗体benchmarkの公式README](https://github.com/piercelab/antibody_benchmark)はbound/unbound構造を
事前重ね合わせしたことを明記している。独立centeringだけではorientation情報は消えない。
今後その入力を使う場合は両partnerへ独立のseed固定SO(3)変換を施し、評価だけがその変換を復元する。
current pilotのraw実apo PDBを使う入力とは別の注意点である。

既存poseのDockQ値、今回のoracle診断、次回のblind予測を別の表に保存する。
学習loss／checkpoint選択は現案にはなく、物理定数・準備・探索hyperparameterの選択履歴を残す。
cofolding全盛の中で目指す価値は、既知のapo入力から、接触力・変形方向・内部コスト・clash解消を
一貫して追跡できる予測器である。その価値も精度優位性も、比較実験なしに成立したとは言わない。

## 5. 追加候補と見送りの理由

Top候補だけ相互作用曲率 `H_int v` をHVPで求め、`K+H_int` のtrust-regionで応答方向も更新する案は、
現在のR2vが元方向の振幅だけを縮小する制限を緩める。全並進の二階微分FFTは不要。
ただしH_intは不定になり得るため、既存SPD solverへ無条件に加えられない。
同じ実energy/force呼出し予算の一階局所最適化と比べ、追加最適化回数の差を有効性と混同しない。
機構試験ではR/tと候補poolを固定し、全armをa=0から始める。R2v、追加一階最適化、
曲率trust-region、元応答方向のspan内だけで同じ曲率を使うradial対照の4条件を比較する。
full対radialが方向補正の効果、full対一階法が曲率計算へ予算を使う価値を測る。
候補選択後の精密化は全armで同じ予算・ゼロ変形開始にし、HVP回数と実時間はgradientと別に記録する。

局所曲率からentropy補正を作る案はさらに後とする。
調和・単一basin・正定値性・物理unitが必要で、現scoreにlog-detを足すだけでは自由エネルギーにならない。
[FFTΔG原著](https://pmc.ncbi.nlm.nih.gov/articles/PMC5834390/)はFFTと自由エネルギー計算の接続の先例だが、
低分子での結果をこのprotein–protein系の改善根拠にはしない。

詳細な一次資料・公開実装・ライセンス・再集計command：

- [接触物理](research/accuracy_contact_physics.md)
- [内部変形と誘導応答](research/accuracy_internal_response.md)
- [探索と評価](research/accuracy_sampling_evaluation.md)

外部codeの取り込み・外部docking engineの実行は今回行っていない。全repositoryを網羅したという調査ではない。
