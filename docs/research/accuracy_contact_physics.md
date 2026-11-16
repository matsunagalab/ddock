# 接触距離と物理相互作用場による予測精度改善の調査

2026-11-16。調査時 ddock HEAD `de42131`。担当：accuracy_contact_physics。
本メモはコード・原著・公開repositoryの調査と、小さなXML／数式の監査である。
新しいdocking run、DockQ評価、nativeを使う係数選択は行っていない。網羅調査ではない。

最優先案は **ff14SBの固定Lennard–Jones（LJ）パラメータから、ligandのprobe型ごとの
受容体potential gridを作ること**。相互作用の接触距離を定義し直し、その同じ場をJVPで変形応答へ接続する。
現在の2 Å格子に差し替えるだけでは不十分で、連続pair potential→格子・位相→実apoの順に反証する。

## 1. 確認できた現在の問題

- `fields.py` は occupancy `1-exp(-density)` とexpanded occupancyの差をsurfaceにする。
  既定pilotのshape項は `3∫C_R C_L − ∫S_R S_L`。原子半径はGaussian幅の定義に使われ、
  pair potentialの零点や極小には直接対応しない。
- §9.19の連続円柱積分ではC–O極小6.120413 Å、幾何半径和3.22 Å。
  cross項へ置き換えただけの案も5.173526 Åで不採用。これは既存の実測で、今回の新測定ではない。
- `chemistry.py:169` はOpenMM `NonbondedForce` からchargeだけを取り出す。
  同じforceにあるsigma／epsilonを保持せず、`GEOMETRIC_RADII_ANGSTROM` を別途使っている。
- `sparse_fields.py` の局所stencilとscatterは同じGaussianを計算する省メモリ処理で、接触モデルそのものは変えない。
- current energyはeffective ε単位。Amber電荷が入っていることは、全scoreがAmberのkJ/molになっていることを意味しない。

原子ごとの幾何半径和を全化学種の実験平衡距離と同一視するのも不適切。
LJ案のoracleは **同じff14SBパラメータから計算するLJ零点・極小・力** にする。

## 2. 公開実装・一次資料からの示唆

| 確認対象 | 実際に読んだ箇所／license | 利用できる考え方と限界 |
| --- | --- | --- |
| LightDock | [cpydock driver](https://github.com/lightdock/lightdock/blob/master/lightdock/scoring/cpydock/driver.py)、[C energy](https://github.com/lightdock/lightdock/blob/master/lightdock/scoring/cpydock/energy/c/cpydock.c)、[GPL-3.0](https://github.com/lightdock/lightdock/blob/master/LICENSE) | charge・vdW・desolvationを明示的に分ける。C実装はLJの上側clip、距離依存誘電、SASA由来contact solvationを使う。clipをそのままJVP/Hessianへ移すのは不適切。AMBER94型対応であり、ddockのff14SBとの差もある。 |
| ATTRACT | [grid_calculate.cpp](https://github.com/sjdv1982/attract/blob/a45f6e086a42dc3f45e28e0c83a361466d79ba96/bin/grid_calculate.cpp)、[nonbon8.f](https://github.com/sjdv1982/attract/blob/a45f6e086a42dc3f45e28e0c83a361466d79ba96/bin/nonbon8.f)、[GPL-3.0](https://github.com/sjdv1982/attract/blob/master/LICENSE.txt) | 型ごとのpotential／gradient格子を保持する実例。nonbon8には8–6／12–6、pair-type係数、electrostatic、switchがある。格子化と物理型を両立する先例だが、その係数や線形switchをコピーしない。 |
| AutoDock Vina | [potentials.h](https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/src/lib/potentials.h)、[Apache-2.0](https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/LICENSE) | Vinaのrepulsionは半径和からの距離、AD4項は型別potential。型別mapは低分子dockingの確立した実装パターン。protein–proteinの精度保証にはならず、Vina scoreへ置き換える提案でもない。 |
| OpenMM | [NonbondedForce API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html)、[標準力の式](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#lennard-jones-interaction)、[license](https://github.com/openmm/openmm/blob/master/docs-source/licenses/Licenses.txt) | LJ、Lorentz–Berthelot則、C2 switchingを独立oracleに使える。API／Reference／CPUはMIT、CUDA／OpenCLはLGPLと明記される。 |
| AGBNP / GaussVol | [gaussvol.cpp](https://github.com/Gallicchio-Lab/openmm_agbnp_plugin/blob/f1c5eaa19012be07cc61375b8d3ec22a8fdfa4da/gaussvol/gaussvol.cpp)、[Reference kernel](https://github.com/Gallicchio-Lab/openmm_agbnp_plugin/blob/f1c5eaa19012be07cc61375b8d3ec22a8fdfa4da/platforms/reference/src/ReferenceAGBNPKernels.cpp)、[LGPL-3.0](https://github.com/Gallicchio-Lab/openmm_agbnp_plugin/blob/master/LICENSE) | Gaussian overlapから分子volume、surfaceと微分を計算する先例。READMEの最終試験環境はOpenMM7.2.2で、現8.6.1との互換性は未検証。研究室websiteの大括りなGPL表記より、実repo LICENSEのLGPL-3.0を採った。 |

LightDockで読んだblob SHAはdriver `4c63960d1487ba5c854ba90603865cd477cbed4a`、
C energy `3c3df9f53d216d939f351973800ba28a3dfe430b`。
GitHub connectorを使用。Vinaのfetch_fileだけreadable-file errorになり、公式raw URLをwebで確認した。
今回は外部コードの取り込み・依存追加は行っていない。

[pyDock原著](https://pubmed.ncbi.nlm.nih.gov/17444519/)はelectrostaticsとdesolvationの重要性、
[iATTRACT原著](https://pubmed.ncbi.nlm.nih.gov/25402278/)はinterface flexibilityと剛体配置の同時精密化の先例になる。
[GaussVol原著](https://pubmed.ncbi.nlm.nih.gov/28160511/)ではvolumeの原子半径微分からsurfaceを求める。
したがって「力場grid」「柔軟精密化」「Gaussian solvent geometry」自体をddock独自とは主張しない。
独自性候補は、**同じ物理fieldの微分をFFTへ載せ、配置選択の時点で有限変形の実energyを検証する接続**にある。

[PIPER原著](https://pubmed.ncbi.nlm.nih.gov/16933295/)の少数相関への分解は参考になるが、
DARSのようなnative由来統計potentialを採用する案ではない。
[HDOCK原著](https://pubmed.ncbi.nlm.nih.gov/28521030/)のtemplate/hybrid戦略を再現する計画でもない。
FFTという汎用計算技術を共有することだけで手法全体が同じにはならない。

## 3. 案A：固定LJ型のpotential grid＋同じ場の変形応答（優先）

各原子の `sigma_i`、well depth `e_i` を、現在使うOpenMMの同じsystemから抽出する。
長さÅ、energy kJ/molへ変換し、原子identityとparameter hashを保存する。

\[
 \sigma_{it}=(\sigma_i+\sigma_t)/2,\quad e_{it}=\sqrt{e_i e_t},\quad
 V_{it}(r)=4e_{it}[(\sigma_{it}/r)^{12}-(\sigma_{it}/r)^6].
\]

連続式の零点は `sigma_it`、極小は `2^(1/6)*sigma_it`、深さは `-e_it`。
既存のGaussian shell厚から間接的に接触距離を決める必要がない。
ただしこれはLJ成分の性質で、電荷・溶媒を含めた総energyの極小をそこに固定する意味ではない。

受容体のprobe型tに対して `R_t(x)=sum_i V_it(|x-r_i|)` を作り、
ligandを同型の原子mass-assignment `L_t(g)=sum_{j:type=t} w_g(r_j)` で格子へ載せる。
`sum_g w_g(r)=1` としたdimensionless mass weightsなら、`sum_t,g R_t(g)*L_t(g-tau)` がenergyで、
ここへ既存fieldの `h^3` を重複して掛けない。density表現を採るなら `rho=L/h^3` として積分係数を一度だけ掛ける。

固定回転で全並進をこの相関から求められる。受容体方向B_mに対してJVP `dR_t/da_m` を作れば、
現在のresponse searchへ接続可能。ligandはC2 cardinal cubic B-splineなどのcompact assignmentを候補とする。
受容体potentialを正確にsampleし、両側へ幅広Gaussianを掛ける追加のdouble smoothingは避ける。
B-spline assignment自体にも平滑化誤差があるため、continuum oracleと比較する。

### 深い衝突で発散させず、接触極小を変えないC2候補

これは本調査での **未実装の設計案**。dimensionless `u=(r/sigma_it)^2`、`u_c=0.9^2` とし、
`h(u)=4(u^-6-u^-3)` をu_c以下だけ二次Taylor polynomialで置き換える。

\[
 \widetilde h(u)=h(u_c)+h'(u_c)(u-u_c)+\tfrac12h''(u_c)(u-u_c)^2\quad(u<u_c).
\]

`V=e_it*h`。r²の多項式なので原点でも有限・滑らか、joinで二階微分まで一致する。
`r>=0.9 sigma_it` は元のLJと同じで、零点・井戸を動かさない。
内側のh'は負なのでr>0では外向き反発、r=0だけは滑らかな放射対称関数に必然の零gradientになる。
全重なりを許可する意図ではなく、大きなenergyと別のclash検査で扱う。
遠方は、接触領域より外の固定switch-on/off（例えば10/12 Å）でC2 taperを掛ける候補。
0.9と10/12 Åは提案値で、apo結果を見て選んだ値ではない。

### 実装時の重要な制約

1. **現2 Å格子を流用しない。** 鋭い接触をresolveできるか、h=1/0.5/0.25 Åとphaseで先に確認する。
   0.25 Åを全protein格子に使えるという見積りではない。原子対fixture上の収束試験である。
2. 型別potentialを増やすと、既存3channelから最大13 LJ＋electrostaticへ増える。
   M方向のderivative cacheと格子細分化の両方がmemoryを増やす。必要解像度で費用が大きければ、
   coarse FFTは候補提案と位置付け、同じ元pair potentialの局所direct検証・精密化を使う。
   coarseの候補取りこぼしをdirect精密化では救えないので、recall診断を別途行う。
3. current effective ε priorへkJ/molのLJを無係数で足すのは単位不整合。
   全atomistic strainと溶媒を含めるか、明示したenergy unit E0で全項を無次元化する。
   ANM gamma／regularizerの物理unitも同じ定義に合わせる必要がある。
4. 真空Coulomb＋LJだけでは埋没電荷を過大安定化し得る。LJ差し替えをbinding free energyモデルの完成とは呼ばない。
   固定誘電／screened kernelは探索surrogateとして明示し、GB/PBの検証と区別する。
5. intramolecular 1–2／1–3除外と1–4 scalingはcross-chain pairとは別。
   strainへ拡張するときはOpenMM exceptionsも保存し、すべての原子対へ同じ式を掛けない。

## 4. 案B：compact C2 vdW球のunion（安価な幾何baseline）

原子半径Rの内外で薄い幅deltaだけ滑らかに接続するindicator `b_i` を使い、
`C=1-product_i(1-b_i)` とする。bは内側1、外側0、境界はquintic smoothstep。
元のGaussianの長いtailを切り、Rを実際の境界へ近付ける。
core–core overlapとcore–shell crossなら3channel相関を維持できる。

ただし **compactにするだけで原子対の正しい接触極小が保証されない**。
shell厚、delta、repulsion/attraction係数、原子数に伴うunion飽和の影響が残る。
surface方向fieldのdot productを使う案も考えられるが、追加のenergy係数とgrid勾配が必要。
この案は低channel数のgeometry baselineであり、LJ／溶媒モデルではない。
§9.19のcross案失敗を踏まえ、native-free pair／plane試験を通らなければproteinへ進めない。

優先度はAより低い。Aが必要とする格子・メモリが過大だった場合のcheap proposerとして比較する価値がある。
実apo DockQに合わせてRやshell厚を調整すれば、物理に基づくという主張が弱くなる。

## 5. 案C：接触に伴う脱溶媒和と変形応答を結ぶ（Aの次）

接触面が増えるだけで常に得をするscoreから、極性を埋める費用も考えるscoreへ進める。
簡単なnonpolar continuumでは `G_np=p V + gamma A` のcomplex-minus-monomersを評価する。
同じsmooth union `C_RL=C_R+C_L-C_R C_L` に対し、
`Delta V=-integral(C_R C_L)` は相関なのでFFTに乗る。
しかし `A=integral|grad C|` の面積差は一般には双線形でなく、正確なsurface energy全体が
少数FFTになるとは主張できない。極性GB項のBorn半径も相手が来ると変わる。

実装の現実的な順序は、Aによる候補poolへ固定parameterのGB/SASAまたは既存OBC2/ACEを用い、
interactionとmonomer strainを分けて実energy検証すること。その後にsurface／Born環境のJVPを
近似応答へ加え、実complex energyで受理する案を検討する。
GaussVolは微分可能なgeometryの先例だが、現在のOpenMMとの互換性は未確認で即時依存にはしない。

§9.19のOBC2再採点はR1 Top1を改善したがDockQ0.23未満、R2の改善は一貫しなかった。
したがって「再採点だけで解決」は否定的な既存証拠がある。
この案の差分は探索のcontact物理を先に直し、非線形変形と脱溶媒和を一緒に検証すること。
抗体loopでは極性groupの埋没と相手依存変形を扱える余地があるが、現時点の予測精度は未測定。

## 6. 小さな実測：ff14SB型数とC2候補の式

ローカルの `protein.ff14SB.xml` をPython標準XML parserで読んだ。
同ファイルはcurrent chemistryの `FORCEFIELD_FILE` と対応。
SHA256 `d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8`。
`(sigma,epsilon)` の異なる組は14、そのうち1組はepsilon=0。
従って **このff14SB全体でも非零LJのprobe型は最大13**。
実apo pairに含まれる型の数はまだ抽出しておらず、13は上限である。
残基名・原子名ごとに数百channelを作る必要はない。

連続C2候補の式を同じPythonで算術評価し、u_c=0.81で
`h=6.6361189533`、`h'=-77.0331244413`、`h''=768.9653510375`、
原点の延長energy `h_tilde(0)=321.2920331585` を得た。
これはgrid実装・力誤差・精度の検証ではなく、有限coreの大きさを確認する計算のみ。
`r_min/sigma=1.1224620483`。係数選択はこの出力から変更していない。

乱数・学習loss・checkpoint・train/test splitはなし。CPU上の標準Python scalar/XML処理で、
シミュレーションなし、skip／失敗なし。実行command：

```bash
python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
p=Path('.venv/lib/python3.12/site-packages/openmm/app/data/amber14/protein.ff14SB.xml')
root=ET.parse(p).getroot()
vals=sorted({(float(a.attrib['sigma'])*10,float(a.attrib['epsilon'])) for a in root.findall('NonbondedForce/Atom')})
print(hashlib.sha256(p.read_bytes()).hexdigest(),len(vals),vals)
uc=.9**2
h=lambda u:4*(u**-6-u**-3)
hp=lambda u:4*(-6*u**-7+3*u**-4)
hpp=lambda u:4*(42*u**-8-12*u**-5)
extension=lambda u:h(uc)+hp(uc)*(u-uc)+.5*hpp(uc)*(u-uc)**2
print(extension(0),h(uc),hp(uc),hpp(uc),2**(1/6))
PY
```

補足：Lorentz–Berthelotの `(sigma_i+sigma_j)^n` を二項展開すると、
`rho_m=sum_i sqrt(e_i)*sigma_i^m delta(x-r_i)`、m=0..12の13momentでLJ係数を厳密分離できる。
12次の13項と6次の7項はligand側rho_mを共有でき、receptor側potentialへまとめられる。
これは本調査での代数的導出で、新規性は未判定。
ただし今回の13非零型に対してchannel削減にならず、pair-scaled C2 coreは共通radial核でなくなり、
この単純分離がそのまま成立しない。共通kernel regularizationは極小やdynamic rangeに別の難点がある。
従って最初の実装には採らず、型別gridを正確性の基準にする。

## 7. 最初の反証可能な実験（まだ未実行）

問いは「Aがnative情報なしに接触距離・力を修正し、FFT/JVPと両立できるか」。
13非零LJ型の全unordered91組を対象とし、同一元素でも異なるparameter組を省略しない。
0 LJ型は別にenergy/force=0の検査を行う。targetなしの物理unit testであり、成功する化学種だけ選ばない。

1. **連続oracle**：固定型対のLJとC2延長について、零点・極小・極小値・force符号・join二階微分を確認。
   `r>=0.9sigma` で元LJとの差をfloat64精度内にし、0<r<0.9sigmaで反発が反転しないこと。
   tiny-r、完全重なり、cutoff端点、epsilon=0も含む。
2. **格子oracle**：同じ小さな単原子fixtureをh=1/0.5/0.25 Å、各軸phase=0/h/2で比較。
   FFT整数shiftと同一grid direct sumは1e-9相対／absolute floor付きで一致すること。
   連続式との極小距離誤差はh<=0.5で0.1 Å以内を暫定gate、forceは井戸近傍の絶対誤差も記録。
   閾値は実行前に登録し、失敗時に結果に合わせて緩めない。
3. **JVP／非線形**：原子移動に対するfield JVPと中心差分を比較。grid cell境界、softcore join、cutoffを通る。
   元の全原子kinematicsから複数方向の応答を出し、実pair energyによる検証と同じ単位か確認。
4. **費用判定**：実apoのinput-derived grid／全型／M方向で必要byte数を先に見積もる。
   h=2を便宜的に許可せず、解像度gateで必要なspacingを使う。高価ならcoarse proposer＋direct verifierを比較案にする。
5. **次の精度実験へ**：上記gateを満たしてから、同一回転・candidate budget・固定poolで旧scoreとAを比較。
   最初はrigidのみで接触score変更の効果を分離し、その後に同じscoreのinduced-fit armを追加。
   開発済みbarnase/barstarはmechanism診断用と明記し、未使用apo集合を別途固定する。
   oracle DockQを使ってrotationや係数を選ばず、候補recallと最終rankingを別に評価する。

結論：Aは現在測定された接触距離の欠陥を直接検証でき、物理parameterとFFTを結び付けやすい。
Bは低コストgeometry baseline、Cはbinding物理の不足を補う次段階。予測精度改善はいずれも未確認であり、
まずAの連続・格子・微分gatesを通すのが、現時点で最も小さく反証可能な前進である。

## 8. pose coverage診断の独立review

担当Cの `scripts/diagnose_inducedfit_pose_coverage.py` と、評価専用の
`data/inducedfit/evaluator_only/pose_coverage_v1/` および `pose_coverage_score_v1/` を読んだ。
目的はapo→native→受容体frameの合成、quaternion規約、元の探索domainとの対応を確認すること。
予測モデルやparameterを変えず、元のdocking／DockQ計算も再実行していない。
以下は実行済みのread-onlyチェックを記録したもの。本節追記時の再計算は行っていない。

**確認結果。** macOS arm64、ddock `.venv` のPython3.12.11／NumPy float64／Torchを使用。
合成座標はNumPy local generator seed20261116の17点。新しい訓練・loss・checkpoint選択なし。
native-derivedな評価probeを読むため、予測精度の独立testやnative-free実験とは扱わない。

| 検査 | 結果 |
| --- | ---: |
| 保存128 quaternionのrow-vector行列対 `geom.rotate`、最大成分差 | 4.4408920985e−16 |
| 合成座標のKabsch fit再構築、最大成分差 | 2.6645352591e−15 |
| 合成apo→native→受容体frameの逐次変換対合成式、最大成分差 | 8.8817841970e−16 |
| q41／t=[10,2,22]を719 ligand重原子へ適用した座標対oracle PDB、最大原子距離 | 0.000795686722 Å |
| 元受容体PDB対oracle PDB、座標差 | 0 Å |
| DockQで選ばれる原子数（受容体／ligand） | 864／719、推論と一致 |
| DockQの選択原子identity・順序対保存済み推論mapping | 両鎖で完全一致 |
| DockQ環境の実version | 2.1.3 |

最初の単純ATOM行比較で受容体872行と表示されるのは、8個のalternate location重複による。
追加のofficial parser確認では864原子が選ばれ、SER28 CB/OG、SER38 CB/OG、ILE96 CB/CG1/CG2/CD1は
いずれもaltloc Aで、推論の選択と一致した。ligandにaltlocはない。
この入力についての座標変換・原子対応にblockerは見つからなかった。全チェックの失敗は0。

**解釈の限界。** q41／t=[10,2,22]は保存DockQで0.3677056784、acceptable分類だが、
fnat=0.2、fnonnat=0.738095、heavy clash=3、apo-fit回転との差30.5155°である。
元scoreが+1150.368827 ε、元R0 Top5 cutoffが−84.101241 εという差だけでは、
「物理的に正しい構造を誤って罰した」と断定できない。nativeに近い向きであっても、
剛体apoの衝突や不適切な接触を罰している可能性が残る。
成立するのは、**元の離散探索domain内にDockQ acceptableと判定される評価専用poseが少なくとも1つ存在するが、
元scoreとpeak pickerでは選ばれない**という限定した結論である。
探索成功、物理的妥当性、全poseのDockQ上限、未使用testでの改善は示さない。

また `rigid_direct_energy` はligand座標を直接shiftして場の内積を採点したもの。
`physical_energy` 内の静電potential convolutionはFFTを使うため、
**独立したatomistic pairwise energyとの比較ではない**。
FFT translation相関と同じ固定座標のfield採点の整合性として解釈する。

実行済みの合成座標／quaternion／PDB丸め確認command：

```bash
.venv/bin/python - <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'scripts')
from diagnose_inducedfit_pose_coverage import fit_row, quaternion_row
from zdock.geom import rotate
rng=np.random.default_rng(20261116)
x=rng.normal(size=(17,3))
frozen=json.loads(Path('data/inducedfit/apo_pilot_v1/pilot_run/report.json').read_text())
rotation_errors=[]
for q in frozen['quaternions']:
    xt=torch.tensor(x,dtype=torch.float64)
    actual=torch.stack(rotate(*xt.unbind(-1),torch.tensor(q,dtype=torch.float64)),-1).numpy()
    rotation_errors.append(np.max(np.abs(actual-x@quaternion_row(q))))
q1=rng.normal(size=4);q1/=np.linalg.norm(q1)
q2=rng.normal(size=4);q2/=np.linalg.norm(q2)
rr,rl=quaternion_row(q1),quaternion_row(q2)
tr,tl,cr,cl=rng.normal(size=(4,3))
rfit,tfit=fit_row(x,x@rl+tl)
rrelative=rl@rr.T
trelative=(cl@rl+tl-tr)@rr.T-cr
sequential=(x@rl+tl-tr)@rr.T-cr
composed=(x-cl)@rrelative+trelative
print('seed',20261116,'dtype',str(x.dtype),'quaternion_count',len(rotation_errors))
print('max_quaternion_vs_geom_component_error',max(rotation_errors))
print('max_synthetic_fit_component_error',np.max(np.abs((x@rfit+tfit)-(x@rl+tl))))
print('max_synthetic_composition_component_error',np.max(np.abs(composed-sequential)))
def coords(path,chain):
    rows=[line for line in Path(path).read_text().splitlines() if line.startswith('ATOM') and line[21]==chain]
    return np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in rows])
coverage=json.loads(Path('data/inducedfit/evaluator_only/pose_coverage_v1/report.json').read_text())
probe=coverage['nearest_angle_probe']; rr41=quaternion_row(frozen['quaternions'][probe['rotation_index']])
cr=np.array(frozen['preparation']['receptor']['center']);cl=np.array(frozen['preparation']['ligand']['center'])
lig=coords('data/inducedfit/apo_pilot_v1/1A19_A_protein_only.pdb','A')
rec=coords('data/inducedfit/apo_pilot_v1/1A2P_B_protein_only.pdb','B')
pred_lig=(lig-cl)@rr41+np.array(probe['grid_translation'])+cr
oracle='data/inducedfit/evaluator_only/pose_coverage_v1/nearest_angle_grid_translation.pdb'
print('oracle_ligand_atom_count',len(lig),'max_heavy_export_error_A',np.linalg.norm(pred_lig-coords(oracle,'L'),axis=1).max())
print('oracle_receptor_atom_count',len(rec),'max_receptor_export_error_A',np.linalg.norm(rec-coords(oracle,'R'),axis=1).max())
assert max(rotation_errors)<1e-12
assert np.max(np.abs(composed-sequential))<1e-12
assert np.linalg.norm(pred_lig-coords(oracle,'L'),axis=1).max()<.001
PY
```

実行済みのofficial parser／推論atom mapping照合command：

```bash
data/inducedfit/dockq_env/bin/python - <<'PY'
from DockQ.DockQ import load_PDB
from pathlib import Path
import json, numpy as np
r=json.loads(Path('data/inducedfit/apo_pilot_v1/pilot_run/report.json').read_text())
for side,orig,chain,outchain in [('receptor','data/inducedfit/apo_pilot_v1/1A2P_B_protein_only.pdb','B','R'),('ligand','data/inducedfit/apo_pilot_v1/1A19_A_protein_only.pdb','A','L')]:
    a=load_PDB(orig,chains=[chain])[chain]
    b=load_PDB('data/inducedfit/evaluator_only/pose_coverage_v1/nearest_angle_grid_translation.pdb',chains=[outchain])[outchain]
    atoms=list(a.get_atoms());outs=list(b.get_atoms())
    print(side,'DockQ_selected_atoms',len(atoms),'oracle_selected_atoms',len(outs),'inference_selected_atoms',r['preparation'][side]['heavy_atoms'])
    print('DockQ_altloc_selection',[(int(x.parent.id[1]),x.name,x.get_altloc()) for x in atoms if x.get_altloc().strip()])
    mapping=r['preparation'][side]['chemistry']['heavy_atom_mapping']
    expected=[(z['atom_key']['residue_number'],z['atom_key']['insertion_code'],z['atom_key']['residue_name'],z['atom_key']['atom_name']) for z in mapping]
    actual=[(int(x.parent.id[1]),x.parent.id[2].strip(),x.parent.resname,x.name) for x in atoms]
    print('exact_atom_identity_order_matches_inference',actual==expected)
PY
data/inducedfit/dockq_env/bin/python - <<'PY'
import importlib.metadata
print('DockQ',importlib.metadata.version('DockQ'))
PY
```

結論：今回の幾何診断の変換・quaternion・原子対応は独立確認を通過した。
次に判断すべきなのは、このacceptable幾何probeの高energyが、妥当な衝突罰則、接触modelの欠陥、
または両方によるのかである。成分別energyと原子接触を分け、nativeで係数を調整することなく検証する。
