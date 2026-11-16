# 内部変形と誘導適合：予測精度改善の調査

2026-11-16。調査時のddockは `de42131`。研究ログの正本は `EXPERIMENT_REPORT.md`。
本メモはコード・原著・公開実装の調査と保存済みJSONの再集計であり、新しいドッキング計算、
最適化、nativeによる係数選択は行っていない。以下の新方式は未実装・未評価の提案である。

## 結論

内部変形側の第一優先は **分子内nonbondedを含む変形コストと基準状態の整合**。
次に、現在の一律の小さい振幅球を **候補ごとの原子変位制約** に置き換える。
その上で、既存ANMで表現できない **相手の残差力に応じた局所torsion方向** を追加する。
後二者を先に入れると、誤ったscoreに対して非物理的な変形能力を増やす危険がある。
接触距離scoreの修正は、これらと並ぶ前提条件である。

## 1. 確認済みの実装と測定事実

`src/zdock/inducedfit/modes.py` のANMは方向を選ぶために使われる。
実際の `angle_basis` はanchored CA Jacobianへのdamped least squaresで求め、CA metricを白色化する。
`deformation_energy` は全CA network辺の距離springと振幅二乗regularizerのみ。
Kは同じ非線形写像を含むpriorのHessianなので、ANMの固有値をそのまま使う実装ではない。

`hydrogen_kinematics.py` はN/CA/C rootを固定したφ/ψ回転で水素も動かす。
結合長・結合角を保持するが、Ramachandran、分子内nonbonded、χ回転、loop両端の閉鎖を保証しない。
`relax.py` と `verified_response.py` は同じsmooth intermolecular field＋CA springを評価する。
実CA変位・最大原子変位を検査する一方、原子力場の分子内strainは目的関数に入らない。
verifierは `a=0` でpriorが0、かつ変形strainが非負である契約を持つ。

### 1.1 大きなstrainはtorsion項だけの問題ではない

元の母集団はapo pilot v1の全30構造（R0/R1/R2各5候補の初期・最終）。
`atomistic_obc2_v1.json` は30/30採点済み、failure0。
既に問題が指摘されたR2の3構造はpost-hoc診断例であり、独立testや新しい成功集合ではない。
以下は最終15構造すべての受容体strainを保存値から再読したもの（kJ/mol）。
初期15構造は変形ゼロであり、同じJSONに保持されている。

| arm | selection rank | bonded | vdW | Coulomb | GBSA | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 1–5（全5件） | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| R1 | 1 | 0.346212 | 2.813514 | 2.275156 | 2.257772 | 7.692654 |
| R1 | 2 | −0.840726 | −0.275216 | 11.177858 | 0.701375 | 10.763291 |
| R1 | 3 | −0.462981 | −1.196264 | 4.245411 | −0.025635 | 2.560530 |
| R1 | 4 | 1.582308 | 13.756254 | −10.006229 | 2.554218 | 7.886551 |
| R1 | 5 | 1.491421 | 17.778844 | −14.836437 | −1.114162 | 3.319666 |
| R2 | 1 | −3.548477 | 81.497822 | 54.011579 | −3.572154 | 128.388769 |
| R2 | 2 | −4.117739 | 120.764533 | 13.087842 | 2.929703 | 132.664339 |
| R2 | 3 | 0.346212 | 2.813514 | 2.275156 | 2.257772 | 7.692654 |
| R2 | 4 | −3.797081 | 50.917803 | 38.113627 | −1.135853 | 84.098496 |
| R2 | 5 | 1.491421 | 17.778844 | −14.836437 | −1.114162 | 3.319666 |

R2 selection1/2/4ではbonded成分はむしろ減り、vdWとCoulombが増えている。
したがって「torsionポテンシャルの追加だけ」でこの問題を解く根拠は弱い。
ただし、これはH-only準備・未最小化apo・固定protonation・OBC2/ACEのpotential energy差であり、
実験的変形自由エネルギーではない。vdW増加だけから個々の衝突原子も特定できない。
また現 `atomistic_score.py` のsevere clashは**分子間**の定義で、0件でも分子内clashの保証にはならない。

### 1.2 ANMを増やすと全方向が動きにくくなる可能性

2-mode pilotと8-mode smokeの保存diagnosticsは次のとおり。

| 指標 | 2 modes | 8 modes |
| --- | ---: | ---: |
| ANM→torsion fit相対残差 | 0.127–0.141 | 既存2方向0.127–0.141、追加6方向0.295–0.449 |
| 最大torsion row norm（rad/Å） | 0.375451 | 3.836889 |
| 最大原子の方向作用素norm（Å/振幅Å） | 3.971432 | 19.755382 |
| 共通原子上限1.5Åから導かれる球半径 | 0.377697Å | 0.075929Å |

最後の行は同じ上限を代入した算術上の比較。8-mode smoke自体は原子上限1Åで半径0.050619Å、
圧縮auditは1.5Åで0.075929Åを使った。半径の異なるsmokeとpilotのaccuracy比較ではない。
全方向を `dmax / max_i ||B_i||op` で制限すると、特定の高レバレッジ方向が全体を約1/5に絞る。
fit残差の増加もあるため、ANM本数を増やすだけで局所柔軟性が改善するとは言えない。

## 2. 案A：原子力場の分子内コストを一貫したpriorにする（第一優先）

**仮説。** CA変形は小さいが分子内vdW/Coulombを大きく悪化させる方向が、現在は過大評価される。
この方向を抑制すれば、相互作用に応じた有用な変形に探索予算を使える。

第一段階は、全候補について新規分子内clash、vdW pair増加、torsion成分、apoにおける残差力を分解する。
新しい拒否閾値を問題3件からfitせず、元全30構造を保持して機構を確かめる。
OpenMM既存systemからparameterとexceptionsを得れば、1–2/1–3除外と1–4 scalingを誤って二重計算せずに済む。
ff14SBは側鎖・主鎖torsionを含む原子力場であるが、単独のtorsion項だけを全変形コストとはみなさない。
[ff14SB原著](https://pmc.ncbi.nlm.nih.gov/articles/PMC4821407/)、
[OpenMMの非結合相互作用](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html)。

第二段階は、単位の揃った物理相互作用scoreと次のpriorを組み合わせる。

```text
U0(θ) = U_ff14SB/OBC2(x(θ)) + U_anchor(x(θ); apo_input)
θ* = apoだけを使った、指定したtorsion自由度・変位領域内の局所最小点
Edef(a) = U0(θ* + A a) − U0(θ*)
K = ∂² Edef / ∂a² at a=0
```

これは単量体ensembleを用意する方式ではなく、全poseで共通の一回のapo準備。
重原子を動かす場合は移動量・準備条件・失敗を記録し、全比較armに同じ準備を使う。
anchorは実験入力への構造保持priorという新しいモデル仮定であり、隠れたnumerical dampingとしない。
CA springへ原子力場を単純加算せず、CA springを置き換える比較とする。
残すanchorの強度はnative-freeな準備変位の制約から事前固定し、DockQで選ばない。

**重要な失敗条件。** 未最小化apoで単純に `U(x(a))−U(x0)` を足すと一般に一次項が残り、負のstrainも起きる。
ε単位の現smooth scoreとkJ/molを無係数で足してはならない。
apo準備がstationaryにならない場合やKが正定値でない場合は、その系を失敗として記録し、
固有値の黙ったclipや負strainの0への丸めで成功にしない。
`U−U0−g0·δθ` と線形項を引く方法はapoを力学的に釣り合わせる局所surrogateとしては考えられるが、
外力による補償を入れた別モデルであり、元Amber potentialではないので第一選択にしない。

FFT探索に必要なのは固定Kと各方向の場のJVPであり、intramolecular prior自体をFFT化する必要はない。
小さいMならOpenMM forceからtorsional torqueを作り、有限差分HVPでKを組める。
`x` をÅ、OpenMM forceをkJ/mol/nmとすると `dU/dθ = −0.1 J_Å^T F_nm`。
kinematic mapの二階微分も必要なので、単に `J^T H_cartesian J` とするより、torque全体を差分する。
OpenMM force接続の微分は独立有限差分で検証する。
[OpenMM PeriodicTorsionForce実装](https://github.com/openmm/openmm/blob/master/openmmapi/include/openmm/PeriodicTorsionForce.h)。

**最初の反証実験（未実行の提案）。** 既存apo受容体・8 modes・全原子準備とhashを固定し、
原点および各方向の±0.01/0.05Å相当振幅で、CA prior、ff14SB/OBC2成分、原子内pair距離、torqueを記録する。
seed0、CPU float64、PyTorch1 thread、OpenMM Reference。新規学習loss/checkpointはない。
gradient差分のstepは1e−4/5e−5 radの2条件を両方報告し、都合のよい方だけ採らない。
apo-only準備の前後で残差torqueとKを確認し、物理priorが非物理的変形方向を抑えないなら仮説を棄却する。
native精度の評価は、この機構確認と接触score検証の後に固定した未使用apo集合で行う。

## 3. 案B：一律の球から、候補ごとの原子変位制約へ

**仮説。** 現在の最大作用素normによる球は、安全だが方向を区別しないため過度に保守的。
物理的に動かせる低レバレッジ方向まで縮めており、柔軟ドッキングの利点を消している可能性がある。

候補poseのFFT勾配gに対して、次の小次元凸問題を解く。

```text
min_a g^T a + 0.5 a^T K a
subject to ||a||2 <= requested_CA_radius
           ||B_i a||2 <= max_atom_displacement  for every atom i
```

KがSPDなら凸QCQP（SOCP表現も可能）。元の保守的球はこの可行集合に含まれるため、
同じ二次モデルなら最適gainは悪化しない。ただし、非線形geometry・energy・DockQの改善は保証しない。
微小変位の制約は実変位ではないので、現verifierのtrue CA RMSD／全原子最大変位／実gainを残す。

全FFT並進に対してこの問題を解くのは避ける。固定quotaのrigid poolと、
CA球だけを使うoptimistic response poolをそれぞれ取り、その和集合で原子別問題を解く。
両poolを残すのは、変形の過大利得だけで候補予算を埋めないため。
例として各100候補、重複を除き最大200候補、Top10、同じ128回転を**結果を見る前に**登録する。
この値は新しい推奨既定ではなく、計画時に固定すべき試験予算の例である。

最初の実験は新しいdockingではなく、既存raw response poolの全候補を共通集合として
球制約／原子別制約のgain、可行性、active原子、実変形gain、物理strainを比較する。
同じK/g/B、同じ要求CA半径と原子上限を使い、数値精度と全失敗数を確認する。
球と同等か、実検証で全て縮小されるなら、この方向の精度改善余地は小さい。
大きなgainと同時に原子力場strainが増えるだけなら採用しない。

これは制約最適化の適用であり、原子別制約そのものの新規性を主張しない。
ddock固有の研究点は、全並進FFTの候補生成と、実原子変位・strainに基づく応答検証の接続にある。

## 4. 案C：既存ANMの外にある接触力への応答を、局所方向として追加

**仮説。** 最低ANM方向を増やすより、現在の相手が加える力のうち既存基底で解消できない成分を
局所torsionへ写す方が、少ないFFT方向でinterfaceの柔軟性を増やせる。
力に対する線形応答自体は既存のPRSの考え方であり、新発明とはしない。
[PRS原著](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758672/)、
[ProDyの実装](https://github.com/prody/ProDy/blob/main/prody/dynamics/perturb.py)。

1. 修正済み物理scoreで第一passを実行し、固定基準で多様な候補clusterを選ぶ。
2. 各代表poseで原子力を求め、`gθ = Jθ^T ∇x Einter` を計算する。
3. 予測接触の周囲で許可したφ/ψ/χへ限定し、物理HessianのHVPによるsolveで応答方向を得る。
4. 既存角度基底AとのH内積重複を除き、独立な最大4方向を追加する。
5. 基底とKと場の微分を再構築し、第二pass中は固定する。

HがSPDで制約なしの局所式なら、`v = −H^(-1)gθ` に対し
`p = v − A(A^T H A)^(-1) A^T H v` とすれば既存方向とH直交する。
拘束付きの真の局所solveはこの式だけでは済まず、局所maskとloop境界条件も含めて解く。
行列Hを保存せずHVPと反復solveを使い、M方向だけ全原子JVPを評価する構成は既存省メモリ経路と両立する。

最初の公平な比較は、固定8 ANM、4 ANM＋4接触応答、4 ANM＋4 random局所方向。
同じ第二pass FFT本数、同じ代表pose数、同じiteration/energy評価数とし、basis構築費用も計上する。
nativeなしで作る方向でも誤scoreの候補へ自己強化する危険があるので、rigid/global候補の固定quotaを残す。
第一passの誤scoreを直さずにこの適応を行わない。

HADDOCK3のflexrefは接触interfaceを段階的に側鎖・主鎖flexibleにする実装例。
iATTRACTも全interfaceの柔軟性と剛体姿勢を同時に最適化する先行研究であり、局所柔軟性自体は既知。
ddockで狙う差は、接触に応じた少数方向を**FFT候補選択へ戻す**ことと、その応答コストを
同じ物理priorと有限変形で検証すること。
[HADDOCK3 flexref実装](https://github.com/haddocking/haddock3/blob/main/src/haddock/modules/refinement/flexref/__init__.py)、
[iATTRACT原著](https://pubmed.ncbi.nlm.nih.gov/25402278/)。

## 5. 抗体と両partnerへの拡張

現在のopen-chain φ/ψ写像を局所CDRにそのまま切り出すとC末端側anchorが動く。
loop端点・向きの拘束Jacobian Cのnullspaceに角度方向を置き、有限step後はclosureへretractionする必要がある。
線形nullspaceだけで有限closureが保証されたとはしない。χ自由度も加え、主鎖を曲げて側鎖衝突を避ける
誤った代償を減らす。disulfideを含む抗体に進む前に、現準備のcrosslink拒否を明示的に改修する。

KGSはnonlocal constraints、nullspace move、衝突回避、exact loop rebuildingの公開実装を持つ。
これはddockの微分可能なmapへそのまま移植できる証拠ではないが、endpoint constraintsと有限closureを
分ける実装の参考になる。HBを全て固定すると誘導適合に必要な組み替えを妨げるため、
CDRのanchor/disulfideと柔軟な非結合contactを区別する。
[KGS原著](https://pmc.ncbi.nlm.nih.gov/articles/PMC4517867/)、
[NullspaceMove](https://github.com/ExcitedStates/KGS/blob/master/src/moves/NullspaceMove.cpp)、
[ExactIK](https://github.com/ExcitedStates/KGS/blob/master/src/loopclosure/ExactIK.h)。

ligandも変形させる拡張では、まずreceptor/ligandそれぞれの一次微分場を使うblock-diagonal応答とし、
両者の交差曲率はTop候補の局所refinementで計算する。全探索へ交差項を入れるとM_R×M_L個の相関が増える。
partnerを入れ替えた同条件実験も必要で、片側だけ変形するモデルの非対称性を誤差から分離する。

## 6. 確認した公開コードと再利用の範囲

2026-11-16にGitHub connectorで以下のファイルを確認した。網羅的な全repository比較ではない。
コードのコピー、依存追加、外部engineの実行は行っていない。

| repository | 確認した参考点 | ライセンス確認 |
| --- | --- | --- |
| OpenMM | PeriodicTorsionForce、既存ddockのff14SB/OBC2経路と力の接続 | 参照headerにMIT形式の許諾。repository全構成要素の一律ライセンスとは断言しない |
| ProDy | `dynamics/perturb.py` のcovariance×forceによるPRS | [LICENSE.rst](https://github.com/prody/ProDy/blob/main/LICENSE.rst)でProDy本体MIT、同梱依存には別記載 |
| KGS | nullspace、clash-aware move、ExactIK | [LICENSE](https://github.com/ExcitedStates/KGS/blob/master/LICENSE)の許諾はMITに類似するpermissive条文。GitHub SPDXはNOASSERTIONなのでMITと断定しない |
| HADDOCK3 | `flexref`のinterface side-chain/backbone refinement | [LICENSE](https://github.com/haddocking/haddock3/blob/main/LICENSE)はApache-2.0。CNSは[別条件](https://www.bonvinlab.org/haddock3-user-manual/install-cns.html) |
| LightDock | [structure/nm.py](https://github.com/lightdock/lightdock/blob/master/lightdock/structure/nm.py)でCA ANMを全原子へ拡張、両partnerのANM設定 | [repository](https://github.com/lightdock/lightdock)はGPL-3.0。比較対象・設計参照としコード取込なし |

## 7. 今回の再集計の再現と限界

保存値の読取りのみ。新たなloss、seed、checkpoint、hardware比較、parameter selectionはない。
元の計算条件はff14SB/OBC2/ACE、OpenMM Reference、NoCutoff、pH7の固定protonation、
apo development1系。H-only準備、原子力場でのapo最小化なし。今回native構造は開かなかった。

入力hash：

- `atomistic_obc2_v1.json`: `74a43787d601a329a8993a4ef33052e85562972c0e50b27c42126fabbf0b0176`
- `apo_pilot_v1/pilot_run/report.json`: `e7d1c14e55e8b2c992a62b6df3489b9934dfc17467414f322ce68f39453dd16e`
- `anm_memory_v1/runner_smoke/report.json`: `658a3510941699bf0a981baca26e4036fc357136f81cc7fa6672e124cd545679`

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path('data/inducedfit')
r = json.loads((root/'apo_pilot_v1/atomistic_obc2_v1.json').read_text())
print('all rows', len(r['rows']), 'failures', len(r['failures']))
for row in r['rows']:
    if row['stage'] == 'final':
        s = row['score']['receptor_strain']
        print(row['arm'], row['selection_rank'],
              {k: s[k] for k in ['bonded', 'vdw', 'coulomb', 'gbsa', 'total']})
for run in ['apo_pilot_v1/pilot_run', 'anm_memory_v1/runner_smoke']:
    r = json.loads((root/run/'report.json').read_text())
    print(run, r['mode_diagnostics'], r['effective_radius'],
          r['maximum_atom_motion_per_unit_amplitude'])
PY
```

**結論・限界・次実験。** 保存済みの失敗例には、分子内nonbondedコストの不足と、
モード増加に伴う全方向の保守的な制限という二つの具体的な改善余地がある。
ただし、どちらの変更もaccuracy改善は未測定。まずnative-freeなforce/strain整合と原子別可行性を確かめ、
その後、接触scoreを直した同条件のapo実験で予測精度を比較する。局所接触応答とloop閉鎖は次段階とする。
