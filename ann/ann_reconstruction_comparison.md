# ANN Reconstruction Flow Comparison

本文档总结并比较以下两个脚本中的 ANN 重构流程与方法细节：

- [reconstruct_pantheon_ann_unweighted_binnedcv.py](/home/geng/Codes/sgl_cosmo/ann/reconstruct_pantheon_ann_unweighted_binnedcv.py)
- [rec_dd_ann.py](/home/geng/Codes/sgl_cosmo/SGL_gamma0711a/rec_dd_ann.py)

## 1. `reconstruct_pantheon_ann_unweighted_binnedcv.py`

### 1.1 整体目标

这个脚本的目标不是直接对强透镜样本做拟合，而是先用 Pantheon+ 超新星数据重构一条平滑的 `m_B(z)` 关系，再把该关系及其导数在指定红移 frame 上输出，并通过 Monte Carlo ensemble 给出均值和协方差。

它当前采用的是：

- `scikit-learn` 的前馈 `MLPRegressor`
- Pantheon+ 的 `m_b_corr`
- full covariance 驱动的 correlated realizations
- 基于 redshift bin 的分层交叉验证
- 基于 bin 容量反比的加权训练与加权验证评分

### 1.2 输入数据与选择

脚本默认读取：

- `ann/data/Pantheon+SH0ES.dat`
- `ann/data/Pantheon+SH0ES_STAT+SYS.cov`

关键输入列是：

- `zHD`
- `m_b_corr`

样本选择上，只保留：

- `IS_CALIBRATOR == 0`

因此它并不直接使用 `MU_SH0ES` 作为训练目标，而是用标准化后的 corrected apparent magnitude `m_b_corr`。

### 1.3 重构目标量

脚本并不是直接让网络去学原始 `m_B(z)`，而是做了一个低红移主导形状的分解：

- 基准形状：
  - `5 log10(z)`
- 基准导数：
  - `5 / (ln 10 * z)`

实际训练目标是一个二维输出：

1. `residual_m = m_B(z) - 5 log10(z)`
2. `residual_dm_dz = dm_B/dz - 5/(ln 10 * z)`

也就是说，它让网络去学一个“去掉低红移对数主项后的残差函数”和对应的残差导数。

这样做的动机是：

- 降低低红移处 `m_B(z)` 的强非线性
- 让网络更容易学习剩余的平滑结构
- 在恢复时再把解析基准项加回去

### 1.4 导数目标的构造

脚本内部会先根据 `m_B(z)` 样本构造导数目标：

1. 按红移排序
2. 对重复红移点先做聚合
3. 用 `UnivariateSpline` 对 `m_B(z)` 做平滑
4. 对样条求导，得到 `dm_B/dz`

然后再减去解析基准导数，得到网络真正学习的导数残差。

因此，这个脚本的训练目标是“函数值 + 导数”的联合学习，而不是只学一个标量输出。

### 1.5 网络结构与模型定义

基础模型由 `build_regressor()` 构造，核心是：

- 输入 `z`
- `StandardScaler` 对输入做标准化
- 主回归器为 `MLPRegressor`
- 输出端再通过 `TransformedTargetRegressor` 对目标做标准化

默认单模型后备设置是：

- `hidden_layer_sizes = (32, 32)`
- `activation = tanh`
- `solver = lbfgs`
- `alpha = 1e-3`

但真正用于正式训练的超参数，不是这组默认值，而是由 CV 从参数网格中挑出的最优配置。

### 1.6 超参数搜索与选模

脚本会先进行一次交叉验证选模。

候选参数网格包含：

- `hidden_layer_sizes`
- `activation`
- `solver`
- `alpha`
- 对 `adam` 还包括 `learning_rate`

这里的交叉验证不是普通 `KFold`，而是：

- 先按 `min(zHD)` 起点和 `--cv-bin-width` 划分 redshift bins
- 用这些 bins 作为标签做 `StratifiedKFold`

如果某些 bin 的样本数少于折数，会先与相邻 bin 合并，保证 CV 可以正常运行。

### 1.7 bin 容量反比权重

当前这版脚本虽然文件名还叫 `unweighted_binnedcv`，但方法上已经加入了基于 bin 容量的权重。

权重函数思路是：

1. 把红移按固定 bin width 分箱
2. 统计每个 bin 的样本数
3. bin 内样本数越少，权重越高

具体形式可以写成：

- 先计算
  - `raw_weight ~ (mean_bin_count / bin_count)^power`
- 再做归一化，使平均权重约为 1
- 再通过
  - `1 + strength * (normalized_weight - 1)`
  控制幅度

因此当前控制这部分的关键参数是：

- `--cv-bin-width`
- `--bin-weight-strength`
- `--bin-weight-power`

这些权重会同时作用在：

- CV 训练阶段
- CV scorer 中的误差项
- realization 正式训练阶段

### 1.8 损失与评分

这个脚本训练时用的是 `MLPRegressor` 自带回归损失，但“选模”用的是自定义 composite score。

CV scorer 的主要组成是：

1. `m_B` 残差的加权 MSE
2. 导数残差的加权 MSE
3. `dm_B/dz` 的高阶曲率惩罚
4. `10^(m_B/5)/(1+z)^2` 的高阶曲率惩罚

其中：

- `derivative_weight` 控制导数误差项
- `curvature_penalty_weight` 控制导数曲率惩罚
- `distance_curvature_penalty_weight` 控制距离形状惩罚

为避免量纲不同导致某一项主导，脚本会先估计训练目标的标准差，并对各项做标准化。

### 1.9 full covariance 与 Monte Carlo realizations

这是这个脚本最重要的方法特征之一。

它不会只用对角误差训练，也不会只训练一个网络，而是：

1. 从 Pantheon+ 的 full covariance 抽取筛选后的子协方差
2. 用 Cholesky 分解构造 correlated realizations
3. 对每个 realization 训练一个单独的 MLP
4. 在目标 frame 上评估所有网络
5. 由 ensemble 的分布给出：
   - `m_b_mean`
   - `m_b_cov`
   - `dm_b_dz_mean`
   - `dm_b_dz_cov`

也就是说，这个脚本的误差传播是：

- 数据层：full covariance
- 模型层：network ensemble
- 输出层：prediction covariance

### 1.10 输出内容

主要输出包括：

- `pantheon_mb_ann_reconstruction.csv`
- `pantheon_mb_ann_reconstruction.npz`
- `pantheon_mb_ann_best_params.json`

其中 `csv` 里核心字段是：

- `z`
- `m_b_mean`
- `m_b_sigma`
- `dm_b_dz_mean`
- `dm_b_dz_sigma`

`npz` 会保存：

- 原始观测和 realizations
- frame 上的 ensemble predictions
- 输出协方差
- 未收敛 realization 的掩码

### 1.11 物理用途

这个脚本本身停留在：

- `m_B(z)` 重构
- `dm_B/dz` 重构

它没有直接输出强透镜距离比表，但可以在后续 notebook 里进一步通过假设 `M_B` 转成：

- `D_L(z)`
- `D_A(z)`

然后再与强透镜或理论模型比较。

---

## 2. `SGL_gamma0711a/rec_dd_ann.py`

### 2.1 整体目标

这个脚本的目标更直接，是：

1. 用 Pantheon+SH0ES 的距离模数训练一个 Keras/TensorFlow ANN
2. 直接预测 luminosity distance 及其误差
3. 再进一步计算强透镜系统需要的：
   - `dd`
   - `D_A_l`
   - `D_A_s`
   - `D_A_ls`
4. 把这些结果直接写回强透镜表

因此它是一个“从超新星训练到透镜表输出”的一体化工作流。

### 2.2 输入数据与训练目标

这个脚本读取的是：

- `Pantheon+SH0ES.dat`

训练目标不是 `m_b_corr`，而是：

- `MU_SH0ES`

并且先把距离模数转换成光度距离：

- `D_L = 10^((MU_SH0ES - 25)/5)`

误差也直接用对角误差传播：

- `D_L` 的误差来自 `MU_SH0ES_ERR_DIAG`

因此它训练的是二维目标：

1. luminosity distance
2. luminosity distance error

这和前一个脚本很不一样：它直接在物理距离空间学习，而不是先在星等空间重构再传播。

### 2.3 数据划分

脚本采用随机切分：

- 80% 训练
- 10% 验证
- 10% 测试

切分方式是对索引打乱后直接划分，不考虑红移分层，也没有交叉验证。

因此它的模型选择和泛化评估更像标准机器学习训练流程，而不是天文统计里常见的 ensemble + CV 重构流程。

### 2.4 网络结构

Keras 模型非常直接：

- 输入维度 1
- 一个隐藏层：
  - `Dense(2048, activation='elu')`
- 输出层：
  - `Dense(2)`

优化器使用：

- `Adam`
- 学习率衰减 `ExponentialDecay`

损失函数实际上写成了：

- `Huber(delta=8000.0)`

评估 metric 是：

- `mse`

因此，这个模型本质上是一个单隐藏层、大宽度的前馈网络。

### 2.5 训练流程

训练阶段：

1. 读入 Pantheon+SH0ES
2. 把 `MU_SH0ES` 转成 `D_L`
3. 随机划分 train/val/test
4. 如果已存在保存好的模型，就直接加载
5. 否则训练 300 epochs
6. 保存模型权重

这个脚本还保留了 `loss_list = ['mae', 'mse']` 的循环结构，但当前实际编译模型时用的是固定的 Huber loss，所以这部分更像历史遗留框架。

### 2.6 误差建模方式

这个脚本输出的第二列是“网络直接学出来的误差”。

也就是说：

- 它没有用 Pantheon+ 的 full covariance
- 没有 Monte Carlo realizations
- 没有 ensemble covariance

误差的来源是：

1. 训练标签中直接包含一个 `D_L` 误差列
2. 网络把这个误差列也当成回归目标一起学习
3. 后续对 `D_A`、`dd` 的误差传播都是基于这列进行代数传播

这是一个非常工程化、直接的做法，但统计上比 full covariance ensemble 的处理要简化很多。

### 2.7 预测到透镜量的转换

脚本定义了两个核心物理接口：

#### 2.7.1 `distance_ratio(zl, zs)`

利用网络预测：

- `D_L(z_l)`
- `D_L(z_s)`

再构造：

- `dd = 1 - (D_L(z_l)/D_L(z_s)) * ((1+z_s)/(1+z_l))`

并给出误差传播。

#### 2.7.2 `distance_DA(zl, zs)`

先把预测的 `D_L` 转成共动距离：

- `D_c = D_L / (1+z)`

然后：

- `D_c,ls = D_c,s - D_c,l`

再转成：

- `D_A,l`
- `D_A,s`
- `D_A,ls`

以及对应误差。

### 2.8 `rec_curve.npy`

脚本还会生成一个独立的重构曲线文件：

- `rec_curve.npy`

生成方式是：

1. 在 `z = 0~3` 上均匀取 `100` 个点
2. 用训练好的网络预测 `D_L(z)` 及误差
3. 转成 `D_A(z) = D_L(z)/(1+z)^2`
4. 保存 `[z, D_A, sigma_DA]`

这也是后续 notebook 中旧 ANN 曲线常常读取的来源。

### 2.9 输出

最终，这个脚本会：

- 保存模型权重
- 保存 `rec_curve.npy`
- 读取强透镜表
- 为强透镜表新增：
  - `dd_ANN`
  - `dd_error_ANN`
  - `da_l_ANN`
  - `da_l_err_ANN`
  - `da_s_ANN`
  - `da_s_err_ANN`
  - `da_ls_ANN`
  - `da_ls_err_ANN`
- 最后输出新的 `SGLTable_ANN.csv`

因此它是一个直接面向 SGL 下游分析的产物。

---

## 3. 两者的核心比较

### 3.1 训练目标的本质差异

`reconstruct_pantheon_ann_unweighted_binnedcv.py`：

- 训练的是 `m_b_corr`
- 更准确地说，是 `m_B - 5 log10(z)` 及其导数残差
- 本质上是在星等空间做非参数重构

`rec_dd_ann.py`：

- 训练的是 `MU_SH0ES` 转换后的 `D_L`
- 直接在距离空间回归

这意味着前者更接近现代超新星重构文献中“先在星等空间重建再传播”的思路，后者更像工程化直接拟合距离。

### 3.2 数据误差处理差异

`reconstruct_pantheon_ann_unweighted_binnedcv.py`：

- 使用 full covariance
- 通过 correlated realizations 传播误差
- 输出预测协方差矩阵

`rec_dd_ann.py`：

- 只使用 `MU_SH0ES_ERR_DIAG`
- 没有 full covariance
- 没有 realization ensemble
- 误差作为第二个回归输出直接学习

前者统计上更完整，后者更简单、更直接。

### 3.3 网络与框架差异

`reconstruct_pantheon_ann_unweighted_binnedcv.py`：

- `scikit-learn`
- `MLPRegressor`
- 多 realization、多模型
- 有 CV 选模

`rec_dd_ann.py`：

- `TensorFlow/Keras`
- 单个宽隐藏层 ANN
- 单模型训练
- 无交叉验证，仅 train/val/test 切分

### 3.4 选模策略差异

`reconstruct_pantheon_ann_unweighted_binnedcv.py`：

- 有系统的超参数搜索
- 有 redshift-binned stratified CV
- 有基于 bin 容量的权重
- 评分中还包含平滑/曲率信息

`rec_dd_ann.py`：

- 没有超参数搜索
- 没有 redshift-aware CV
- 主要靠固定网络结构和固定 epoch 训练

因此前者更强调“统计重构质量 + 选模稳健性”，后者更强调“可用的直接拟合器”。

### 3.5 输出用途差异

`reconstruct_pantheon_ann_unweighted_binnedcv.py`：

- 更像一个上游重构引擎
- 主要输出 `m_B(z)` / `dm_B/dz`
- 后续可转为 `D_A(z)` 供其他模块使用

`rec_dd_ann.py`：

- 更像一个下游应用脚本
- 直接输出 `dd`、`D_A,l`、`D_A,s`、`D_A,ls`
- 直接服务于强透镜表分析

### 3.6 物理与统计上的优缺点

#### `reconstruct_pantheon_ann_unweighted_binnedcv.py` 的优势

- 更接近超新星统计重构的规范做法
- 保留 full covariance
- 有 ensemble，不确定度更可解释
- 有系统化选模
- 在红移稀疏区有更灵活的 reweighting / CV 控制

#### 它的代价

- 代码更复杂
- 参数更多
- 训练成本更高
- 结果更依赖对 scorer 和平滑项的设计

#### `rec_dd_ann.py` 的优势

- 逻辑简单直接
- 训练与使用都很方便
- 直接面向 SGL 应用输出
- 不需要额外 notebook 才能得到 `D_A` 和距离比

#### 它的代价

- 忽略 full covariance
- 把误差当作第二个网络输出去学，统计解释较弱
- 没有 CV 与 ensemble，模型稳定性和系统误差控制较弱
- 训练目标采用 `MU_SH0ES`，会把 SH0ES 定标直接带入

---

## 4. 为什么传统 ANN 很难直接引入 covariance，而当前脚本是怎么做到的

这一点其实是两类方法之间最根本的差别之一。

### 4.1 传统 ANN 为什么很难把 covariance 直接作为输入

像 `rec_dd_ann.py` 这种传统 ANN 工作流，训练时通常是：

1. 每个样本是一对或一组简单输入输出
   - 例如 `z -> D_L`
2. 损失函数是逐样本累加的
   - 例如 MSE、MAE、Huber
3. 默认假设每个样本的误差是相互独立的

这类框架很自然地适合处理：

- 每个样本一条独立误差棒
- 或者每个样本一个对角误差

但不自然地适合处理：

- 一个 `N x N` 的 full covariance
- 尤其是 Pantheon+ 这种跨样本非对角相关误差

难点主要有 4 个。

#### 4.1.1 covariance 不是逐样本局部信息，而是全局耦合信息

对角误差可以理解为：

- 第 `i` 个样本自己的误差大小是多少

但 full covariance 还包含：

- 第 `i` 和第 `j` 个样本会不会一起向上波动
- 某个系统误差模态会不会同时影响一串红移点

这说明 covariance 不是某个单点 `z_i` 的局部属性，而是整个样本集合的全局结构。

因此你很难像输入一个普通特征那样，把“第 i 行第 j 列的 covariance”直接喂给网络，并期望网络自然学会如何传播它。

#### 4.1.2 标准监督学习的损失函数通常是逐点写的

传统 ANN 的损失经常是：

\[
\mathcal{L} = \sum_i (y_i - \hat y_i)^2
\]

这等价于默认协方差矩阵是单位阵，或者最多是对角加权。

如果想严格把 full covariance 写进来，理论上更像要用：

\[
\mathcal{L} = (\mathbf{y}-\hat{\mathbf{y}})^T C^{-1} (\mathbf{y}-\hat{\mathbf{y}})
\]

这时就不是“逐样本独立”的损失了，而是一个全局耦合二次型。

在传统 Keras / TensorFlow 回归脚本里，这当然不是不能写，但会带来几个问题：

- batch 训练会变复杂
- 需要始终按整批样本保留正确顺序
- 如果 train/val/test 被随机切开，子样本的 covariance 处理也更麻烦
- 还要小心数值稳定性和矩阵求逆

所以大多数工程型 ANN 脚本最后会退回到“只用对角误差”或者“直接把误差作为第二输出一起学”。

#### 4.1.3 covariance 的目标不是“输入给网络”，而是“传播到输出不确定度”

很多时候，大家最想要的其实不是“网络看到 covariance 矩阵”，而是：

- 最终重构曲线知道这些相关误差的存在
- 输出的不确定度和协方差也反映这种相关性

如果直接把 covariance 当作额外输入特征，网络也未必就会正确地把它传到最终结果里。因为：

- covariance 是误差统计，不是决定物理趋势的自变量
- 网络学到的很可能只是某种经验性相关，而不是严格的误差传播

所以从统计建模角度看，covariance 更适合通过“样本 realization”进入，而不是作为普通输入特征进入。

#### 4.1.4 单个 ANN 输出一条曲线时，不确定度往往只是点对点误差，不是 full prediction covariance

传统 ANN 即使学了误差列，也常见的是：

- 每个红移点输出一个均值
- 每个红移点输出一个误差

但这通常只是逐点 sigma，并不自动给出：

\[
C_{\rm pred}(z_a, z_b)
\]

也就是预测曲线在不同红移之间的相关性。

而如果你的后续科学分析需要：

- 在不同红移点上做联合比较
- 或把整条曲线继续传播到强透镜距离比

那么 full prediction covariance 往往比单点误差更关键。

### 4.2 `rec_dd_ann.py` 为什么没有真正把 covariance 传进去

`rec_dd_ann.py` 的做法是：

- 从 `MU_SH0ES` 计算 `D_L`
- 再用 `MU_SH0ES_ERR_DIAG` 传播出 `D_L` 的对角误差
- 然后让网络同时学习：
  - `D_L`
  - `sigma_{D_L}`

这意味着它学到的是：

- 一个均值函数
- 一个逐点误差函数

但它没有用到：

- Pantheon+ 的 full `STAT+SYS.cov`
- 跨红移点的相关系统误差
- 由 covariance 驱动的 correlated realizations

所以严格来说，这类传统 ANN 并不是“把 covariance 传到了结果里”，而是“把对角误差近似地当作另一个监督目标”。

### 4.3 `reconstruct_pantheon_ann_unweighted_binnedcv.py` 是怎么解决这个问题的

当前脚本采取的思路，不是让网络直接吃 covariance，而是把 covariance 先转换成一组带相关性的 realization，再让 ANN ensemble 去学习这些 realization。

这是关键。

#### 4.3.1 第一步：保留 full covariance

脚本先读取：

- `Pantheon+SH0ES_STAT+SYS.cov`

然后按照筛选后的样本子集，从总协方差矩阵中抽取对应子矩阵。

这一步确保训练使用的误差结构，仍然是完整的 full covariance，而不是只剩对角元。

#### 4.3.2 第二步：从多元高斯分布生成 correlated realizations

脚本调用 `sample_realizations(...)`：

\[
m^{(k)} \sim \mathcal N(m_{\rm obs}, C)
\]

实现上是：

1. 对 `C` 做 Cholesky 分解
2. 生成标准正态随机向量
3. 左乘 Cholesky 因子，得到带相关性的扰动
4. 加到观测均值向量上

这一步的意义是：

- 每个 realization 都是一整条“可能的 Pantheon+ 曲线”
- 而且这条曲线内部各点的联合涨落，严格遵循输入 covariance

换句话说，covariance 被转成了“函数样本的相关扰动模式”。

#### 4.3.3 第三步：每个 realization 训练一个 ANN

对第 `k` 个 realization：

- 先构造目标 `r(z)` 和导数残差
- 再训练一个单独的 MLP

于是最终不是一个 ANN，而是一组 ANN ensemble。

这组网络的差异，不是随便初始化出来的，而是来源于：

- Pantheon+ 数据在 full covariance 下允许的联合波动

这就把 covariance 的信息真正带入了网络集合。

#### 4.3.4 第四步：在统一 frame 上评估全部网络

脚本把所有 realization 对应的网络都评估在同一个 `z_frame` 上。

于是每个红移点上，不再只有一个预测值，而是有一组 ensemble prediction：

\[
\{m_B^{(1)}(z), m_B^{(2)}(z), \dots\}
\]

更重要的是，不同红移点之间的联合波动也保留下来了。

#### 4.3.5 第五步：直接从 ensemble 计算 prediction covariance

最后脚本用这些 frame-level predictions 计算：

- `m_b_mean`
- `m_b_cov`
- `dm_b_dz_mean`
- `dm_b_dz_cov`

因此，输出的不确定度不是网络“猜”的，也不是单点标签误差回归出来的，而是：

- 由输入 full covariance
- 经过 Monte Carlo realization
- 再经过 ANN nonparametric reconstruction
- 最终在输出 frame 上统计出来的 prediction covariance

这就是当前脚本真正“引入 covariance”的地方。

### 4.4 这种做法的统计含义

这个框架的核心思想可以概括为：

- covariance 不直接作为神经网络输入特征
- 而是先定义观测空间中的概率分布
- 再从这个分布中采样出很多可能的数据 realization
- ANN 只是用来学习每个 realization 对应的平滑函数
- 最终由 ensemble 的分散度恢复输出空间的不确定度和相关性

所以从方法论上说，当前脚本里 ANN 扮演的是：

- 非参数函数逼近器

而 covariance 扮演的是：

- 观测误差的统计生成机制

这两者被清晰分开了。

### 4.5 为什么这比“直接把 covariance 喂进 ANN”更合理

主要有三点。

#### 4.5.1 更符合 covariance 的统计角色

covariance 描述的是观测的不确定度和相关结构，不是物理自变量。

把它用于生成 realization，比把它硬塞进输入层更符合它本来的意义。

#### 4.5.2 更容易得到 full prediction covariance

因为最后你拿到的是整个 ensemble 的输出样本，所以自然就可以计算：

- 逐点方差
- 点与点之间的协方差

这是传统“网络输出一个误差列”很难自然做到的。

#### 4.5.3 更容易和文献中的 Monte Carlo / bootstrap 误差传播框架对接

许多重构文献的方法核心其实都是：

- 先在观测空间做 realization
- 再把 realization 传到函数空间

当前脚本正是这种思想在 ANN 版本下的实现。

### 4.6 一句话总结

传统 ANN 难以直接引入 covariance，不是因为神经网络完全做不到，而是因为：

- covariance 是全局误差结构，不是逐点局部特征
- 标准 ANN 损失通常是逐样本独立的
- 直接把 covariance 喂给网络并不能自然得到正确的输出协方差

而 `reconstruct_pantheon_ann_unweighted_binnedcv.py` 的解决方法是：

- **不把 covariance 当作输入特征**
- **而是把它变成 correlated realizations**
- **再用 ANN ensemble 把这些 realizations 传播到重构结果**

所以它真正实现的是：

- 从观测 covariance 到函数重构 covariance 的传播

这正是它相比传统单网络 `D_L + sigma` 拟合更统计严谨的原因。

---

## 5. 一句话结论

如果把两者放在同一条工作流里看：

- `reconstruct_pantheon_ann_unweighted_binnedcv.py` 更适合做“统计上更严谨的 SN 重构母模型”
- `rec_dd_ann.py` 更适合做“快速、直接、面向强透镜应用的距离预测器”

前者偏向“文献化的重构框架”，后者偏向“工程化的应用脚本”。

如果你的目标是：

- 更可靠地比较不同非参数重构策略
- 更认真处理 Pantheon+ 的相关误差
- 更稳健地诊断高红移形状

那么前者明显更合适。

如果你的目标是：

- 直接给强透镜表补上 `dd` 和角直径距离列
- 快速生成一个可用于后续分析的 ANN 输出

那么后者更直接。
