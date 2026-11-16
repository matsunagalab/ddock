# ddock：FFTを保つ誘導適合ドッキングの実装・評価計画

作成日：2026-11-16

状態：計画と初期数値プロトタイプ。2026-11-16に下記の実装状況を追記。

対象コード：調査時HEAD `ec4584b`。研究履歴の正本は [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) §9。

## 現在の実装状況（2026-11-16）

`src/zdock/inducedfit/` に、reference-free入力型と内容fingerprint、滑らかな3チャネル物理場、
JVP、非周期FFT相関、変位制約付き応答、固定基底での候補探索を実装した。
再現用の [verify_induced_fit.py](scripts/verify_induced_fit.py) も実装済み。
新規73件＋既存FFT回帰2件の **75テストが通過**。seed0/1/2の合成原子系で、FFT対直接energyの
最大絶対誤差は `2.22e-16`、変位上限0.01の9候補すべてで制約を満たした。詳細は研究ログ§9.14。

初期段階はP0/P1/P3の数値基盤として `ff711c9` にcommit・pushした。
その後、PDB重原子の診断用読込、単鎖φ/ψ写像、ANMの内部座標への投影、同じ非線形priorからのK、
姿勢＋内部座標の制約付き精密化を追加した。継続分を含む **133テストが通過**。
1UBQ全鎖602原子／76残基の幾何・微分と、先頭4残基＋合成プローブのFFT→精密化接続を検証した。
詳細は研究ログ§9.15。これはapo予測精度の評価ではない。

さらにOpenMM ff14SBによる全水素・部分電荷、結合graphに沿う水素の内部座標変形、
reference-freeなPDB pair manifest、R0/R1/R2 runnerと評価用の重原子PDB出力を追加した。
実1UBQは602重原子＋629水素で前処理が通過し、重原子座標を保った。詳細は研究ログ§9.16。
[run_induced_fit_pair.py](scripts/run_induced_fit_pair.py) の使い方と制限は
[pair runner手順](docs/inducedfit_pair_runner.md) に記録した。

NN訓練のactivation checkpointingも局所逆伝播へ転用した。合成41原子×2のCPU同条件3反復で、
最終energy／amplitudes／PDBを変えず、peakRSS中央値64.46%減、wall時間19.88%増。
FFT/JVP探索の経路は維持する。詳細は研究ログ§9.17。
全水素入力・checkpoint・既存FFT回帰を含む最終対象suiteは **169テスト通過**。
実apo入力の事前登録監査では、PINDER train辞書順5件の適格pairは0/5だった。
サイズ・非標準残基・末端OXT欠損を記録し、成功例が出るまで選別する方法にはしていない。

その後、結晶接触Zn除去と未解像末端の扱いを明示した実apo barnase/barstar 1系で、
128回転・Top5・R0/R1/R2を実行し、独立DockQ評価まで完了した（研究ログ§9.18）。
最終Top1はR0=0.024711、R1/R2=0.024723、最終best5はR0=0.199266、R1=0.201863、R2=0.157074。
**全候補がDockQ0.23未満で、誘導適合の優位性は未確認**。R2は選択時Top1を改善したが、
最終energy順位が不正解姿勢を選び、R1の相対的に良い候補も失った。
応答利得も過大評価し、一例で予測−45.208に対し実−23.641だった。
入力・計算・scorerの失敗は0で、189テスト通過。計算の完了と構造予測の成功を区別する。
次の優先課題は、nativeを使わないtrust region／相互作用曲率の診断、格子・回転密度の収束、
物理energyによる順位付けの改善である。単一開発系への調整を未使用testの性能とは扱わない。

続いて、Top50応答提案の実変形／実energyを検証してからTop5に絞るR2vを実装した（§9.19）。
全50件で振幅縮小により実gainと制約を確認できたが、最終Top1=0.024723、best5=0.157074で
**精度改善はなかった**。過大な応答利得を使う欠陥の修正と、ドッキング精度の成功を分ける。
凍結候補の2/1 Å・half-cell格子診断でTop1は不変だった。
固定ff14SB/OBC2再採点はR1 Top1を0.201863へ改善したが0.23未満で、R2では改善が一貫しない。
CA springにない内部変形コストと、shape項の接触距離が次の優先課題になった。
[v2 protocol](docs/apo_pilot_v2.md) と研究ログに全失敗／null結果を保存した。
無電荷原子対の連続積分では、C–Oの幾何半径和3.22 Åに対して現shape項の極小は6.1204 Å。
core–shell cross項へ置き換えるだけでも5.1735 Åで、十分な修正にならず採用しなかった。
積分刻み半減の差は全4原子対で最大0.000162 Åなので、遠方極小は格子誤差だけの問題ではない。
次はcompactなvdW境界または固定Amber原子型potential gridを単純系で検証し、
接触距離・反発・力・格子収束を確認してから次のapo protocolへ進む。

Ramachandran／内部非結合相互作用の妥当性、実データ母集団の監査と正規評価、
HDOCK比較、抗体closureは残っている。下記の将来CLI群は未実装で、現在使える診断CLIは
`verify_induced_fit.py`、`verify_internal_modes.py`、`verify_internal_fit.py` と上記pair runner。
各段階全体の完了とは扱わない。

実構造で、無正則化のtorsion fitによる過大角度と、CA振幅だけを制限した応答の過大利得が見つかった。
角度fitの明示的Tikhonov dampingと、最大原子変位に合わせた保守的な振幅半径を追加した。
失敗・修正前後の全条件を研究ログに残し、改善したtoy energyをprotein性能の証拠にしない。

```bash
.venv/bin/python -m pytest tests/test_inducedfit_data.py tests/test_inducedfit_fields.py \
  tests/test_inducedfit_response.py tests/test_inducedfit_search.py
.venv/bin/python scripts/verify_induced_fit.py \
  --out data/inducedfit/verification/synthetic_radius001.json \
  --seeds 0 1 2 --threads 1 --radius 0.01
.venv/bin/python scripts/verify_internal_modes.py \
  --pdb data/inducedfit/inputs/1UBQ.pdb --chain A \
  --out data/inducedfit/verification/1UBQ_internal_modes_checked.json
.venv/bin/python scripts/verify_internal_fit.py \
  --pdb data/inducedfit/inputs/1UBQ.pdb --chain A \
  --out data/inducedfit/verification/1UBQ_fragment_fit_bounded_checked.json \
  --residues 4 --iterations 8 --bound-linear-atom-motion
```

現プロトタイプは正しさを優先し、分子場の格子shapeに対して相関FFTを各軸 `2n−1` に拡張する。
以下のメモリ概算のVは**FFTの実際の格子点数**で読み替える必要があり、入力場のVと混同しない。
大きいタンパク質への省メモリ化・GPU速度は未検証。

## 1. 研究の狙いと採用する方針

**相手分子によって生じる変形と、その変形に必要なコストを、FFT探索の候補選択前に考慮する。**
apo入力から始め、剛体のままなら衝突して落とされる正しい配置を、小変形の応答予測で救えるかを検証する。
HDOCKの再実装や、単量体ensembleを増やすことを主題にはしない。まず物理に基づくモデルだけで実施する。

研究仮説は次の三つに分ける。

1. 同じ物理モデル・候補数で、**選択後だけの柔軟精密化より、選択前の応答予測が正しい配置を多く残す**。
2. 予測された緩和利得が、実際の非線形変形で再現され、最終的な順位と構造精度の改善につながる。
3. 相手の接触力から少数の変形方向を追加すると、固定した低周波モードだけでは不足する局所変形を扱える。

最初の主研究は1と2。3は次段階とし、抗体CDR-H3へ展開する。予測されるのは結合に応じた平衡構造の
局所的な緩和であり、結合の実時間経路や速度論的な induced fit／conformational selection の識別ではない。

成功の主判定は、事前固定した未使用apo/flexible集合における、同一入力HDOCKとのTop-1成功率のpaired差。
HDOCK超え、新規性、計算高速化はいずれも未実証である。

## 2. FFTと両立する計算の骨格

### 2.1 一回の探索中は基底を固定する

単量体のapo構造から、変形写像 `x(a)` とその初期Jacobian `B = dx/da|0`、剛性 `K` を準備する。
これは「動ける方向と動きにくさ」の準備であり、相手を見ずに最終的な変形量を決める処理ではない。
初版では、FFTの固定側である受容体だけを柔軟にする。動かす側は入力の定義で決まり、holoから選ばない。

配置 `q=(R,t)`、振幅 `a` に対して、小変形モデルを

```text
E(q,a) ≈ E0(q) + g(q)^T a + 1/2 a^T K a
g(q) = ∂Einter(q,a)/∂a | a=0
```

とする。無制約なら `a* = -K^-1 g`、緩和利得は `-1/2 g^T K^-1 g`。
実装では変位制約付きの解を使う。エネルギーは小さい方が良いという符号に統一する。
Kは変形priorのHessianであり、配置依存の界面Hessianは省略している。
これは相互作用の一次近似＋priorの二次近似であって、全energyの完全な二次Taylor展開ではない。

物理場を `Fc(x(a))` とすると、モード方向の微分場は

```text
Dmc = JVP(Fc ∘ x, a=0, direction=em)
gm(R,t) = Σc wc Corr(Dmc, Lc(R))(t)
         = Σc wc IFFT[FFT(Dmc) × conj(FFT(Lc(R)))](t)
```

で得る。格子体積等の係数は場／重みに一度だけ含める。相関の符号は直接計算で検証する。
**微分場のFFTは回転ループ外で再利用でき、各回転について全並進のモード力を一括計算できる。**
各格子点で全原子最小化やMDを実行する必要はない。

3チャネル、Mモードなら、基本形の回転あたりの計算は ligand側3 FFT と `(M+1)` IFFT。
各モードではチャネル積を先に足してから1回IFFTする。M=4/8から始める。
通常の剛体coreの3 FFT＋1 IFFTより重いが、配置ごとの反復緩和に置き換わる点を測る。
IFFT後の応答補正は格子点ごとの小次元計算で、別の畳み込みにはしない。
制約solverの費用も `O(M × 格子点数 × 二分法反復数)` として別計測し、FFT時間だけで高速性を判断しない。

### 2.2 相手依存の基底更新も、探索の外側に置く

```text
apo由来の基底を固定
  → FFT探索
  → 多様な少数候補の接触力を計算
  → 単量体の剛性で応答を求め、基底を更新
  → 微分場を再計算して固定
  → 次のFFT探索
  → 上位候補の実変形・非線形精密化
```

並進点ごとに異なる基底を作ることは初版では行わない。固定された少数の微分場で全並進を評価する
というFFTの利点を維持する。基底更新回数の初期候補は0回／1回とする。

### 2.3 NN訓練の計算技術から採用するもの

NNの予測器を必須にせず、自動微分・低ランク化・メモリ管理を物理計算に使う。
数学の多くは古典的であり、NN訓練で整備された計算基盤との組合せがここでの利点である。

| 技術 | 本計画での使い方 | 採用時期／制約 |
| --- | --- | --- |
| JVP／VJP | 全座標×全格子のJacobianを作らず、変形方向の場微分を作る。局所段階では原子力を内部座標へ戻す | 最初から採用。scatter・複素演算等の対応を小問題で検証 |
| 適応的な低ランク部分空間 | 接触力に対する物理応答を集め、少数の追加方向へ圧縮する | 固定基底の検証後。GaLoreとの接点は低ランク化の発想であり、optimizerの直接移植ではない |
| HVPと前処理付き反復法 | 多数の二面角でHessianを明示せず応答を解く | 高自由度化後。4–8モードでは小行列の明示分解を優先 |
| implicit differentiation | 緩和構造を含めて、剛性や排除体積の少数係数を較正する | 局所solverが安定した後。hard top-k全体を微分可能としない |
| 分割reduction・選択的checkpoint | 原子数×格子数の巨大中間テンソルを作らず、必要な局所微分だけ再計算する | 最初に分割計算、checkpointはprofiling後 |
| バッチ化・compile・混合精度 | 同サイズ格子や候補をまとめ、場生成と局所緩和のGPU利用率を上げる | 正しさ検証後。FP32基準、小行列・検算FP64。BF16/FP16は初版に必須としない |

JVP/HVPは [PyTorch公式資料](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html)、
適応的な低ランク化は [GaLore, ICML 2024](https://proceedings.mlr.press/v235/zhao24s.html)、
物理に関係する最適化例は [NysNewton-CG, ICML 2024](https://arxiv.org/abs/2402.01868)、
implicit differentiationは [Blondel et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract.html)、
メモリ管理は [PyTorchのselective checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/) を参照。
[KeOps](https://jmlr.org/papers/v22/20-275.html)は分割した微分可能reductionの実装参考とし、初版の依存追加は必須にしない。

SOAP等のNN用前処理器が持つ勾配統計を、タンパク質の物理的剛性と同一視しない。
最適化の高速化と、物理モデルの変更は別のablationにする。

## 3. 既存コードの監査と変更範囲

既存backendはbaselineとして維持し、新規 `src/zdock/inducedfit/` を追加する。
以下はソース読取りで確認した事項であり、実行して再現した障害という意味ではない。

| 現在の箇所 | 確認事項 | 実装方針 |
| --- | --- | --- |
| `score.py` | 二値PSC、nearest、固定surface/core分類があり、trilinear指定だけでは全項が座標微分可能にならない | smooth coreを独立実装 |
| `search.py:docking_search` | `trans_per_rotation` は既にある。既定値1、複数指定でも単純top-kで空間的NMSではない | 回転／並進のconventionを参考にし、候補多様性の規則を全arm共通化 |
| `search.py:refine_poses_gradient` | 最適化中と最終rescoreで補間経路が異なる | 新coreの局所solverへそのまま流用しない |
| `dataset.py:PreparedProtein` | inputに `native_lig/q_star/t_star` が同居 | referenceを持たない推論専用型へ分離 |
| `prep_cache.py` | keyはsystem IDのhash。入力内容・core・modeのhashを含まない | 新規cacheとversionを用意。旧cacheを暗黙流用しない |
| `geom.py:generate_grid` | ligandのx方向extentを使うpadding | 回転不変bounding radius、変位上限、場supportを使う固定GridSpec |
| `eval_search_apo.py` | `--manifest`なし、`--limit`は先頭抽出。受容体固定export、no_grad区間 | 任意manifest・両鎖座標export・明示的autograd区間を持つ新runner |
| 同apo runnerのcheckpoint読込 | alpha／iface／rhoを読む一方、他係数の完全復元は別途監査が必要 | 最新QP baselineの全parameterを監査し、未対応fieldはfail fast |
| `score_decoys_with_pinder.py` | 公式scorerが欠損IDのpenalty行を補う | 登録manifestへleft joinし、正確な対象集合を分母に再集計 |

§9.2の「各回転で1並進」は既定設定の説明であり、機能が存在しないという意味には修正する必要がある。
この訂正は研究ログへ追記し、以前の記述を消さない。

### 新規moduleと責務

| ファイル（実装予定） | 責務 |
| --- | --- |
| `src/zdock/inducedfit/types.py` | `DockingInput`, atom/residue/chain key, frame変換, `GridSpec`, `CandidateSet` |
| `src/zdock/inducedfit/prepare.py` | 内容hash付きmetadata、reference-free入力契約。力場電荷・実tensor cacheは今後 |
| `src/zdock/inducedfit/pdb_input.py` | PDB重原子・atom対応・明示電荷mappingまたはzero診断、既知crosslinkの拒否 |
| `src/zdock/inducedfit/kinematics.py` | 単鎖の主鎖二面角写像、結合幾何・Pro環の保持 |
| `src/zdock/inducedfit/fields.py` | smooth field、同じ離散energyの直接計算、FFT、JVP |
| `src/zdock/inducedfit/modes.py` | apo ANM、内部座標への投影、基底正規化、同じpriorからのK |
| `src/zdock/inducedfit/response.py` | 変位制約付き二次応答、予測利得、valid mask |
| `src/zdock/inducedfit/search.py` | rigid／response探索、共通候補保持、provenance、時間計測 |
| `src/zdock/inducedfit/relax.py` | 非線形energyによるR,t,a同時最適化、geometry判定、両鎖export |
| `src/zdock/inducedfit/adaptive.py` | 次段階：接触力からの基底更新と応答チャネル圧縮 |
| `scripts/build_induced_fit_manifest.py` | eligible pool・利用履歴・split・hashの固定 |
| `scripts/calibrate_induced_fit_core.py` | 選択した場合のみ実装：trainの固定decoy poolで少数係数を較正し、固定configを出力 |
| `scripts/eval_induced_fit_apo.py` | config＋推論manifestで実行。referenceへアクセスしない |
| `scripts/compare_registered_runs.py` | 公式per-decoy結果を登録母集団に集計、paired CI、失敗内訳 |

## 4. 物理モデルを一つに定義する

### 4.1 smooth field core

最初のモデルは、滑らかな排除体積、表面接触、遮蔽静電相互作用の3項とする。
全原子力場／厳密な溶媒自由エネルギーではなく、物理に動機づけられた粗いエネルギーである。
単位をeffective energy unit `ε` とし、kcal/molとは呼ばない。

```text
ρA(x) = Σi exp(−|x−xi|²/(2σi²)) × C2_taper(|x−xi|/σi)
oA(x) = 1 − exp(−ρA(x))
σi = rvdW,i / sqrt(2 log 2)
sA(x) = oA,expanded(x) − oA(x)

Einter/ε = wrep h³/v0 Σx oR(x)oL(x)
           − h³/v0 Σx sR(x)sL(x)
           + wq ℓ0 h⁶ Σx,y ρqR(x) Kscreen(x−y) ρqL(y)

Kscreen(r) = exp(−κ sqrt(r²+b²)) / sqrt(r²+b²)
```

表面拡張幅 `d=1 Å`、softening `b=1 Å`、`κ=0.1 Å⁻¹`、`v0=1 Å³`、`ℓ0=1 Å`を初期値とする。
表面引力係数は1に固定し、全体scaleの識別不能性を除く。これらは提案値であり最適値ではない。
occupancy用Gaussianはpeak=1の非正規化関数とし、積分正規化する電荷場と区別する。
これにより半径拡張でoccupancyが減少せず、surface場の非負性を保つ。
Gaussianは距離4σまでは保持し、4–5σをC2連続なswitchで0へ落とす。screening kernelも初版は
16–20 ÅでC2 switchして0にする。cutoff拡張による収束検査を行い、範囲外を暗黙切断しない。

電荷場は粒子ごとの正規化を含め、`h³ Σx ρq(x)=Σi qi/e` を保つ。その正規化も微分対象。
電荷はversion固定したOpenMM/Amber ff14SBの前処理で割り当てる。追加H・protonation規則・末端・
disulfide・未対応残基を記録し、重原子座標を保持したことを検査する。初期pH規則は7.0、His等の
variant決定は単量体のみを入力として記録する。既存ZDOCK charge LUTへのsilent fallbackはしない。
未対応化学種は事前eligibilityと実行失敗を区別し、処理可能率を報告する。

初期の数値検証は合成電荷を使う。実タンパク質の検証に進む前に、電荷・atom対応・入力geometryの
契約を固定する。電荷なしの2チャネル版はablationとし、失敗例だけ都合よく切り替えない。

### 4.2 FFT・直接計算・局所微分の関係

正本のenergyは「固定格子上で、実際の座標から作った滑らかな場を積分近似した値」とする。

電荷の二重和は、受容体側に `FR,q=ℓ0 h³ (Kscreen * ρqR)` を事前計算し、ligand側を
`FL,q=ρqL`、重みを `wq h³` とする相関にまとめる。ここで `*` は格子体積係数を含まない
離散畳み込み。このpotential場を含む3チャネルをJVP／FFTに渡し、係数の二重計上を防ぐ。

- 格子点単位の並進では、FFT相関と直接voxel和を一致させる。
- 非整数並進・回転・変形では、実座標から場を再構築し、同じvoxel energyを微分する。
- サンプル済み場のFourier位相移動と、非整数位置の原子からの場再構築は一般には同じではない。
  両者を「完全一致」とは主張しない。
- 任意の別の原子対potentialをFFTの正解oracleに使わない。nonlinear occupancyとは別モデルになる。
- 格子幅・原点・回転に対する誤差は、FFTの正しさとは別に収束検査する。

GridSpecは一つの探索・精密化の間で固定。ligand回転半径、実変位上限、field supportとscreening畳み込みsupportを含めた
zero-paddingとvalid translation maskを使い、周期境界の回り込みを排除する。
局所変形で境界に触れる候補は黙ってenergyを切らず、regridして再評価するか失敗stateを返す。

### 4.3 変形写像と単量体コスト

数値検証では `x=x0+Ba` のCartesian写像を用いる。ただしこれだけで最終全原子構造の正しさを主張しない。
正式なapo評価までに、主鎖φ/ψの内部座標写像を用意する。

1. apoのCAばねネットワークから低周波方向を得る。剛体6モードだけでなく、切断成分由来のゼロモードも検査する。
2. その方向を主鎖φ/ψ Jacobianへfitし、`θ=θ0+A a` のforward kinematicsを作る。
3. peptide ω、結合長・結合角、初期側鎖χは固定する。剛体成分との重複を除くgaugeを固定する。
4. CA Jacobianを `BCA^T BCA/NCA=I` に正規化する。振幅の単位は微小変位でのCA RMSD Åとなる。
5. 同じ写像の距離から `Edef(a)=γ/2 Σij[dij(a)−dij(0)]² + η/2 ||a||²` とし、
   `K=∇²Edef(0)` を計算する。ばねedge・cutoff・重みを固定記録する。

実装では、Jacobianの小さい特異値を単純に逆数化すると過大角度を生んだため、
`σ/(σ²+λfit²)` のTikhonov filterを明示的に用いる。初期既定は `λfit=1e-3 σmax`。
これは基底構築の角度正則化であり、非線形priorのηとは別物。fit残差、角度係数、CA metric、
有限変位を併記し、既定値が最適だとは主張しない。

初期ANM cutoffは12 Å、γの単位は `ε/Å²`。ηはゼロモード対策を含む明示的priorで、初期値は
正の剛性固有値中央値の `10^-4` 倍を目安に設定・記録する。数値安定化だけの名目で応答側にのみ追加しない。
基底変換後に元ANM固有値をそのままKとして使わない。`Edef'(0)=0` を検証する。

disulfide・crosslink・鎖切断の拘束を壊す写像は採用しない。対応不能なら理由付き失敗とする。
正しい結合長／角を保ってもRamachandranの妥当性や内部clashは自動では保証されないため、
受理判定にそれらの検査を入れる。検査を連続energyへ取り込む場合は、baselineを含めモデルversionを更新する。

### 4.4 制約付き応答と過大な緩和利得の防止

CA尺度で正規化した基底に対して

```text
min_a  g^T a + 1/2 a^T K a      subject to ||a|| ≤ δ
```

を全並進点で解く。Kを一度固有分解し、固有基底で `aj=−gj/(κj+λ)`。
無制約解が範囲内ならλ=0、それ以外はベクトル化した二分法でλを求める。
初期trust radius候補は `δ=0.5/1.0 Å`。

**採点は、得た制約付き解を `g^T a + 1/2 a^T K a` に代入した値で行う。**
振幅だけclipして無制約の利得を足す誤実装を禁止する。M本のg mapを保持してtranslation tile単位で解くか、
再計算との交換を選ぶ。制約付きでも1本のscalar mapだけへstreamできるとは仮定しない。

RMSD上限と最大原子変位上限は別物。初期の最大重原子変位上限は3 Åとし、linear mapの保守的boundと
実座標変位の双方を確認する。非線形写像で超えた場合はstepを縮める。
実装済み `PreparedResponseReceptor.bounded_radius()` は `max_i ||B_i||₂` から、
`δeff=min(δ, dmax/max_i||B_i||₂)` を求める。同じδeffを選択前応答と比較armの精密化へ渡す。
これはlinear変位の保証なので、非線形写像後のCA RMSDと最大原子変位も確認する。
初期の実構造断片検証はdmax=0.5 Åで行った。何も指定しない既存prototype APIは振幅上限のみなので、
実データrunnerへ進む際はこれらの制約を設定の必須項目にする。
深い重なりから生じる偽の利得は、入力だけで計算するclash gate、変位上限、非線形再評価で検出する。
gate閾値はtrain/validationで固定し、落とした候補数も記録する。

固定Kのdeterminantは配置間で一定なので、この応答補正を配置依存のエントロピーや絶対結合自由エネルギーとは呼ばない。

## 5. 候補保持と非線形精密化

候補保持の変更が誘導適合の効果と混ざらないよう、rigid／responseに同じ回転格子・valid mask・
多様性規則を使う。初期設定はHopf nside=3、spacing=1.2 Å、global pool上限1500、正式な精密化候補数50。
候補保持は回転ごとに最大4並進ピーク、中心間距離3 ÅのNMSを出発点とし、最終clustering規則も共有する。
既存ddockの設定は別途そのまま再現し、この変更の単独効果を測る。

主因果比較では、rigid選択／response選択後の両方を **同じ `a=0` から** 同じ非線形solverへ渡す。
responseの `a*` をwarm startに使う条件は別armにする。変形・剛体自由度と反復予算を一致させる。
初期solverはline search付きL-BFGSを候補とし、SO(3)の局所更新、trust region、geometry受理を明示実装する。
現プロトタイプでは、制約・失敗状態を直接検証するため、projected gradient＋backtrackingを先に実装した。
同じモデルのR1/R2で同じsolverを使う。L-BFGSへの変更は速度・収束を測る別比較とする。
初期上限は候補ごと50 iteration、energy／gradient評価回数上限も250で制限し、実際の呼出数を記録する。
収束判定・失敗時fallbackはvalidation前に固定する。

最終順位は同じ非線形 `Einter+Edef` で付ける。採点後のgeometry修復で座標が変わったら再採点し、
修復前スコアを使い続けない。受容体・ligand双方の最終座標と元atom keyの対応を保存する。

片側変形の限界が確認された場合は、入力を入れ替えた2回の探索を候補各25、合計50にまとめる案を検証する。
rigid側も同じ2回探索とし、時間増分を含める。両側の全モードを各回転で作り直す方法は初版には採用しない。

## 6. 実装段階と進行判定

| 段階 | 実装／成果物 | 次へ進む条件 |
| --- | --- | --- |
| P0 入出力・baseline | reference-free型、manifest、cache、両鎖export、最新QP parameter監査、既存／HDOCK再現用設定 | nativeが推論に届かない。登録IDと失敗分母を再現でき、出力atom対応と公式scorer roundtripが通る |
| P1 smooth core | 3場、直接energy、FFT、JVP、固定GridSpec | 下記数値gate合格。grid依存と境界誤差を定量化 |
| P2 変形と精密化 | ANM、内部座標写像、K、非線形prior、R,t,a solver | `a=0`一致、geometry保持、微分とKが同じenergyに対応 |
| P3 応答FFT | 全並進g map、trust solver、候補保持、provenance | 直接変形との差分検査とKKT検査に合格。M=0/δ=0でrigidに一致 |
| P4 開発比較 | train/validation、R0–R3、機構診断、時間・VRAM測定 | R2−R1の候補改善と最終順位を確認。設定と予算を固定 |
| P5 最終評価 | config hash固定、未使用holdoutを一度実行、全失敗を集計 | paired結果・CI・coverageを報告。勝利もnullも同じ形式で記録 |
| P6 拡張 | 適応基底／応答圧縮、その後H3 closure | 追加計算費用込みで固定基底を上回ること。別の検証設計を事前固定 |

P0–P3は実装の依存順。P4でresponseが改善しなければ、P5へ大規模投入せず、物理core／基底不足／
近似誤差／順位づけのどこが原因かを分ける。testを開いて修正を繰り返さない。

### 数値検証：性能ベンチマークより先に固定する

| テスト | 初期合格基準 |
| --- | --- |
| FFTと同じ離散モデルの直接和 | CPU float64：scale調整absolute `1e-10`、relative `1e-9` |
| JVPとenergy微分 | 全fieldを±変位で再構築。差分幅 `1e-2/1e-3/1e-4 Å` の収束、relative `1e-4` |
| 内部座標・SO(3)の微分 | 同様に完全再構築したenergyで方向微分を確認。零勾配近傍はabsolute誤差を使う |
| 応答solver | KKT relative residual `1e-7`、radius違反 `1e-6`以下（float64） |
| 退化条件 | M=0、δ=0、g=0、極大剛性でrigid極限。strain非負、補正利得≤数値許容値 |
| frameと境界 | identity quaternion `[0,0,0,1]`、逆変換、正負並進、非立方格子、wrapを検査 |
| chemistryとgeometry | 電荷総和、atom key一意性、重原子保持、結合幾何、キラリティ、crosslink |
| 格子依存 | spacing `1.2/0.9/0.6 Å`、原点0／半セルずれ、回転変更によるscore／force／順位の変化を測る |

float32/CUDAの許容値はfloat64参照から設定し、同じ閾値を無条件に課さない。
既存 `tests/test_search.py` の形式を参考に、`tests/test_inducedfit_*.py` を追加する。
実装をなぞるだけのtestではなく、直接oracle、有限差分、制約残差を独立した根拠にする。

## 7. データ・漏洩防止・評価対象

### 7.1 manifestをコードより先に確定する

`input_manifest.jsonl` はapo座標、配列、chain/atom対応、入力hash、事前eligibilityだけを持つ。
native path、DockQ、apo→holo変形量は別の `evaluation_manifest.parquet` に保存し、推論workerへ渡さない。
可動領域・モード・charge・pivot・crop・初期loopに正解界面やholoを利用しない。

開発はPINDER公式trainの両鎖apoが揃うeligible poolから、interface clusterを跨がないfit64／validation64を
seed=0で固定する。候補数を満たせるかは未確認。足りない場合は実数に縮小し、合成変形を数値／機構検証に
補助使用する。最終集合を開発へ移して件数を満たさない。合成変形の結果を実apo性能の代わりにしない。

最終候補はPINDER-XLの両鎖apo集合。既報の342件から既使用PINDER-Sの93件を引いた249件は、
**未使用性を監査する前の候補数**である。dataset版、membership、過去の学習・手動調査・parameter tuning、
homolog/interface overlapを確認し、利用履歴を含むmanifestを作る。DB5.5は既往の開発利用があり、
無条件に新規testとは扱わない。未使用集合を確保できなければ探索的評価と明記する。

### 7.2 apoとflexibleを分ける

評価専用の変形量Dを次で定義する。

1. native複合体で、相手鎖heavy atomから5 Å以内にheavy atomを持つ残基をinterfaceとする。
2. apoとholoの対応CAを使い、各単量体を独立に全対応CAで剛体整列する。
3. interfaceの対応CAの残差二乗を両鎖でpoolし、その平方平均根をDとする。

初期登録案は `D<1 Å`、`1≤D<2 Å`、`D≥2 Å`。主評価は `D≥1 Å` の集合とする。
これは研究固有の層別であり、公式CAPRI difficulty分類とは呼ばない。atom mapping欠損を報告し、
Dが定義できない系はall-apo集計へ残して別層にする。閾値を最終結果を見て動かさない。
主集合が小さすぎる場合は検出力不足を報告する。
各鎖のnative-interface CA対応率80%以上かつ対応CA数5以上をDの定義条件として初期登録する。
不足時に残存部分だけで小さいDを付けず、D未定義層に置く。鎖別coverageと欠損位置を保存する。

全apoを副評価として必ず示し、同じIDのholo入力は別の対照実験にする。holoでの改善だけを主目的の達成に数えない。

## 8. 比較arm・parameter選択・統計

### 8.1 最初の比較

| Arm | 候補選択 | 精密化 | 何を分離するか |
| --- | --- | --- | --- |
| B0 | 現行ddockの正規設定 | 最新QP等の既存手順 | 現行手法からの実用的改善 |
| B1 | 同一apoのHDockLite | 固定した公式実行条件 | HDOCKとの主比較 |
| R0 | 新smooth core、rigid | 同coreのrigid | 新coreの基準 |
| R1 | R0と同じrigid候補 | flexible、a=0開始 | 選択後の柔軟化の効果 |
| R2 | response、同じ候補数 | R1と同じ、a=0開始 | **候補選択前に誘導適合を考慮する効果** |
| R3 | 選択時だけ排除体積を弱める | R1と同じ | 単なる衝突緩和との違い |
| R2w | R2 | response振幅から開始 | warm startの効果、追加ablation |

主機構比較はR2−R1。R2−R0だけでは候補選択の効果を示せない。
R2の純response選択を先に検証し、rigidとの混合を導入する場合は、全50件のうちrigid割当を
0／10の二候補だけでvalidation選択する。混合と純responseを区別し、unionで予算を増やさない。

初期parameter候補はM=4/8、δ=0.5/1.0 Å。γはtrain上で剛性尺度を確認して基準γ0を定め、
`γ0/3, γ0, 3γ0` をvalidation候補とする。深いclash gate・side変形規則・NMSを含む設定を保存する。
基準γ0の決定規則はtrain pilotで固定し、最終結果から逆算しない。

clash gate、NMS、core、solver等をtrain pilotで固定してから、validation探索は基本12設定、
混合割当を含め最大24設定に制限する。選択順はflexible層Top-1 acceptable、同点ならall-apo Top-1、
同点なら失敗込みの実行時間が短い設定、小さいM、小さいδ、最後にconfig ID辞書順とする。
主条件はこの規則で選んだ純R2または混合R2の一つだけ。R2wは副解析で、主条件の選択には入れない。
validationを見た方式変更や追加探索が必要なら、開発履歴へ記録して新versionを作る。
そのvalidationを未使用の独立検証とは呼ばず、最終holdoutは開かない。

物理coreの `wrep,wq` を較正する場合はtrain decoyの固定poolを用いる。候補対の正解DockQ順序に対し、
`log(1+exp((Ebetter−Eworse)/τ))` と正の係数に対するlog-space prior penaltyを使う。
τとprior尺度はtrain energy尺度で定める。positiveを含まない系も探索成績の分母に残す。
同じ較正済みcoreをR0–R3で共有し、NNは訓練しない。coreをjoint calibrationする案は後続の別実験にする。
DockQを使う係数選択は**正解ラベルを用いたsupervised calibration**であり、NN不要という意味である。
`wrep=wq=1` を初期物理仮定とする未較正版R0/R1/R2も開発比較に残し、物理設計と係数較正の寄与を分ける。

baselineとして、LightDockのANMまたはiATTRACTの柔軟精密化を少なくとも一つ追加し、
「剛体HDOCKより良い」だけで既存flexible dockingへの優位を主張しない。
cofolding由来の初期構造を使う比較は別armとし、入力情報量を揃える。

HDockLiteのbinary/version/hash、コマンド、template経路の有無、restraint、候補数を保存する。
PINDER既報のHDOCK値と今回の異なる母集団を直接paired比較しない。web serverのhybrid版と混ぜない。

### 8.2 候補数と時間の二つの公平性

第一表は全方法の新core armでK=50・同じ局所予算を固定する。
第二表は前処理・応答FFT・局所精密化を含む同一walltime制約とする。
rigid側の余剰時間を候補数増加に使う規則をvalidationで固定し、最終集合では変更しない。
cold cache／warm cache、compile初回費用／定常費用を分けて測る。
同一ホスト、CPU thread数、GPU数／型、timeout時に完了済み候補を採用するかを事前固定する。
CPU主体のHDOCKとGPU ddockの速度差をアルゴリズム単独の高速化とは解釈せず、資源構成を併記する。

seedは0を主解析、1/2を頑健性解析とする。決定論的経路の同一計算は水増ししない。
ランダムな基底近似・初期restart等のseedを全arm間で対応させる。

### 8.3 指標と判定

- 主指標：固定flexible集合での公式PINDER Top-1 acceptable-or-better成功率。
- 副指標：all-apo、Top-1 medium/high、Top-5、Oracle@50、連続DockQ、D別の成績。
- 機構診断：選択直後と精密化後のOracle、予測利得／実利得、R1で落ちR2で救う数とその逆、strain、実変位。
- 品質／計算：clash、geometry破綻、coverage、preparation/search/refinement別時間、peak VRAM、CPU RAM。

target単位のpaired bootstrap 10,000回（seed=0）で差の95% CIを求める。
依存するinterface/homology clusterが残る場合はcluster単位にresampleする。
discordant pair数を報告し、target間独立が成立する集合ではexact McNemarも補助報告する。
cluster依存が残る場合はそのp値を主張の根拠にせず、cluster bootstrapを用いる。
poseやseedを独立targetとして数えない。

「HDOCKに勝った」はlocked主指標のpaired差が正で、95% CI下限も0を上回る場合に限定する。
validationの目安として+5 percentage pointsを実用的改善目標に置くが、これは検出力の保証ではない。
Oracleのみ改善した場合は候補選択は有望・順位づけは未解決と結論する。

### 8.4 失敗を分母から除かない

登録したmethod×target全行に結果をleft joinする。OOM、timeout、前処理失敗、候補不足、
geometry破綻、export失敗は別stateで記録し、主成功率では失敗=0。
事前eligibilityは全method共通で固定し、method固有の失敗を後から除外条件にしない。
評価器の不具合は推論失敗と区別し、評価だけ修正再実行する。未解決なら主集計を未完了とする。

公式 `MethodMetrics` の自動leaderboardは対象外IDのpenaltyも混ざり得るため、per-decoy出力を
exact manifestへ限定して再集計する。`--allow-missing` で成功率を高く見せる運用はしない。

## 9. 次段階：FFT用の応答チャネルを相手に合わせて圧縮する

NN訓練の低ランク化から着想を得る研究部分。固定ANMで効果と限界を測ってから着手する。
独自性候補は「相手に応じた変形部分空間を、再利用可能なFFT微分チャネルに変換する」ことである。

### 9.1 接触力に基づく基底更新

1. 第1passの候補から、nativeを使わず、幾何的に多様で深いclashのない配置を最大32件選ぶ。
2. 各配置の物理的な接触力を内部座標へ射影する。
3. 現在の4モードより広い主鎖二面角／拘束接空間で、単量体priorの正則化した応答問題を解く。
   その変位snapshotをtrust regionで制限する。既存の小さいKだけを解いても新方向は生まれない。
4. 基底に含まれない残差を集め、物理metricに沿ってSVD／直交化し、追加方向を4個まで作る。
5. 元の4個のANM方向と合わせた最大8個で、基底・K・微分場を再計算し、第2pass中は固定する。

比較は固定8 ANM、4 ANM＋4適応、4 ANM＋4ランダムの同じ最終次元とし、基底更新費用を含めた
同一時間比較も行う。clash forceが基底を支配する破綻、候補の偏り、更新で失われる解も報告する。
これが改善しなければ「低ランク化だから良い」と解釈しない。

### 9.2 高自由度化した後のチャネル圧縮

格子体積・energy重み・引力の符号を含む受容体側stacked fieldのJacobianをJ、
対応する相手の場をℓと定義すると `g=J^Tℓ`。無制約の応答利得は

```text
ΔE = −1/2 ||(J K^-1/2)^T ℓ||²
```

となる。したがって、**物理剛性で重み付けした場の変化 `J K^-1/2` を少数の方向へ圧縮すれば、
必要なIFFT数を減らせる可能性がある。** 巨大な応答行列 `J K^-1 J^T` は作らない。
JVP/VJP、応答snapshot、必要ならrandomized range approximationを使う。

M=4–8では圧縮の費用が利益を上回り得るため導入しない。多数の二面角を持つ段階で試す。
trust regionがある場合は、縮約後の剛性と変位metricを保持して制約問題を解き直す。
右側の圧縮方向をVとしたとき、変形復元写像 `T=K^-1/2 V` と縮約metric `T^T W T` を保存する。
Wは元座標のRMSD metricで、正規化した初期モードならI。場だけを圧縮して変形座標との対応を失わない。
無制約式をclipしただけのenergyに置き換えない。

ここはGaLoreの直接適用でも新規性の確定でもない。adaptive reduced-order modeling、
Krylov法、既存normal-mode dockingとの関係を追加調査し、貢献を計算量と予測性能で限定して述べる。

## 10. 抗体CDR-H3への拡張

P6では抗原とframeworkの内部座標を固定し、Ab–AgのR,tとH3主鎖φ/ψ、必要な界面χを扱う。
初期loopはKIC等で両端を閉じる。両端の位置・向き、キラリティ、結合幾何を制約として明示する。

微小応答はclosure Jacobianのnull spaceへ投影する。有限step後はretraction／closure solveで戻し、
残差とbranch変更を検査する。離散KIC分岐まで滑らかに微分できるとは主張しない。
全energy Hessianには負曲率があり得るため、高自由度solverは正定値priorのPCGと、
非線形全energyのtrust-region truncated CGを区別する。

大きいH3変形では複数の閉じた初期loopを8/16個試すが、これは局所応答の限界を補う条件として分離する。
ABlooper／cofoldingの初期候補を使う場合、比較法にも同じ候補を渡す。
モデル間のばらつきを単量体のBoltzmann確率とは解釈しない。

SnugDockを比較に入れ、blind dockingと既知epitopeを別々に評価する。
通常のDockQに加え、framework整列後H3 RMSD、closure誤差、strain、内部clash、時間を測る。
抗原homologyと抗体framework/CDR類似性の両方でsplitを監査する。
P5で開いたtestをP6の方式選択へ使った場合、その再評価は探索的結果とし、新方式の確認には別holdoutを用意する。

## 11. 計算資源・実行記録・予定CLI

### 11.1 資源と実行予算の決め方

開発はCPU float64の小問題、性能評価はCUDA GPUを想定する。候補資源はVRAM24–48 GB、
host RAM64 GB程度だが、必要量／十分性は未測定。今回のローカル環境はarm64で、GPUベンチマークは実行していない。
MPS smoke testをCUDAでの速度や精度の証拠にしない。

C=3、M=8の固定側には27本の複素スペクトルが必要。complex64のrFFT配列だけで概算 `108V bytes`、
128³なら約226 MB、256³なら約1.81 GB。これにreal field、M本のgradient map、回転batch、FFT workspaceが加わる。
固定側cacheとautogradの中間保存は別に計上する。

最初はrotation batch=1。trainからサイズ別に16系を固定し、M=0/4/8の時間・VRAMを測る。
grid volumeでbucket化し、OOM時はbatchを1へ減らし、次にtranslation tile縮小／微分map再計算を行う。
Mや物理モデルを特定targetだけ勝手に減らすfallbackは使わない。

```text
Ttotal ≈ Tprep + Nrot × (Tligand_FFT + (M+1) Tcorr_IFFT + Tresponse_solve)
         + K × Tnonlinear_refine + Tadaptive_update
```

この実測後に全実行の時間上限をconfigへ登録する。小・中・大の失敗を含む見積りとし、
現時点で「何時間で終了」「何倍高速」とは約束しない。複数GPUでは独立targetのshardを先に使う。

### 11.2 保存するartifact

| Artifact | 内容 |
| --- | --- |
| `input_manifest.jsonl`／`evaluation_manifest.parquet` | 推論入力と評価referenceの分離、利用履歴、split、hash |
| `run_config.json` | 全parameter、seed、git commit＋dirty diff hash、依存版、dataset hash、hardware、precision |
| `systems.jsonl` | target status、attempt、OOM対処、各段階時間、peak memory、入出力hash |
| `poses.parquet` | 候補由来、rigid score、response gain、振幅、strain、真のenergy、geometry、反復数、座標hash |
| `per_decoy_metrics.csv` | 公式評価器の原結果 |
| `paired_results.csv`／`bootstrap.json` | 全登録targetの比較、resampling単位、CI、主／副指標 |

生成PDB、cache、大規模logをcommitしない。manifest、設定、再現コマンド、集約結果を研究ログへ記録する。

### 11.3 実装予定のCLI契約

**次のCLIは新規実装予定であり、現在は実行できない。** JSON configを使い、未定の依存を増やさない。
パス名は契約案。manifest builderは登録ファイルの上書きを拒み、変更には新versionを必要とする。

```bash
# P0: datasetと過去利用履歴から登録集合を作る
uv run python scripts/build_induced_fit_manifest.py \
  --config configs/inducedfit/data_v1.json \
  --out-dir data/inducedfit/manifests/v1

# P1-P3: 新backendの数値検証（テストも新規実装予定）
uv run python -m pytest tests/test_inducedfit_fields.py \
  tests/test_inducedfit_modes.py tests/test_inducedfit_response.py

# supervised calibrationを採用する場合のみ、trainラベルでcoreを固定
uv run python scripts/calibrate_induced_fit_core.py \
  --config configs/inducedfit/core_calibration_v1.json \
  --fit-manifest data/inducedfit/manifests/v1/fit_training.jsonl \
  --out-config data/inducedfit/calibration/core_v1.json

# P4: reference-free manifestで推論。arm設定はconfigに完全記録
uv run python scripts/eval_induced_fit_apo.py \
  --config configs/inducedfit/r2_validation.json \
  --manifest data/inducedfit/manifests/v1/validation_input.jsonl \
  --run-dir data/inducedfit/runs/r2_validation_seed0 \
  --device cuda --seed 0

# 公式scorerで得た各armのper-decoy結果を登録母集団で比較
uv run python scripts/compare_registered_runs.py \
  --evaluation-manifest data/inducedfit/manifests/v1/validation_evaluation.parquet \
  --runs-manifest data/inducedfit/runs/validation_runs.json \
  --out-dir data/inducedfit/comparison/validation_v1
```

既存CLIとして `scripts/eval_search_apo.py`、`scripts/score_decoys_with_pinder.py` があるが、
現状のapo runnerにはmanifest指定がない。最終holdoutへ `--limit` を代用せず、P0で共通adapterを実装する。
公式scorerの既存CLIは、新export directoryとper-decoy出力を接続して利用する。
較正用manifestはtrainラベルを持つ別契約であり、推論manifestではない。
較正済みcoreの比較では、較正を先に完了してからそのconfigを参照するvalidation推論を実行する。

## 12. 結論・限界・直近の着手順

最初に実装する核は、**JVPで作る少数の微分場＋FFT相関＋変位制約付き応答＋同じenergyによる実変形確認**。
これにより、NNを必須にせず、NN訓練で成熟した計算技術を物理主体の誘導適合へ使う。
FFTを維持する条件は、基底と剛性を一回の全探索中に固定し、更新を探索間に限定することである。

P0–P3の数値接続と限定された力場電荷の入力準備まで進んだ。直近の次工程は、
実データの末端／欠損化学policy・数値収束・baseline契約と、内部非結合コストの追加である。
適応基底、implicit calibration、H3は、固定基底の機構検証を通った後に追加する。
特に大変形、溶媒、側鎖再配置、単量体自由エネルギーの近似は限界として残る。

本計画の初稿作成時には実装・学習・ベンチマークは行っていなかった。
その後の初期実装と合成系の数値検証は冒頭の実装状況および研究ログ§9.14に分けて記録した。
各段階の実施時には、失敗／null結果を含め [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) に累積記録する。
