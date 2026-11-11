# Ames 房价数据：变量中文说明

本文件是原始 `data_description.txt` 的中文对照版。**变量名与类别代码保持原样**，以便直接核对 `train.csv` 和 `test.csv`。面积单位为平方英尺（sq ft）；`NA` 在许多类别中表示该设施不存在，应结合具体变量判断，不要一律当作未知值。

## 房屋类别、位置与土地

- **MSSubClass**：出售住宅的建筑类别。20＝1946 年及以后的一层住宅；30＝1945 年及以前的一层住宅；40＝带完工阁楼的一层住宅；45＝未完工的一层半住宅；50＝完工的一层半住宅；60＝1946 年及以后的两层住宅；70＝1945 年及以前的两层住宅；75＝两层半住宅；80＝错层／多层住宅；85＝分层门厅住宅；90＝双户住宅；120＝1946 年及以后的一层规划住宅社区住宅（PUD）；150＝一层半 PUD；160＝1946 年及以后的两层 PUD；180＝多层 PUD（含错层／分层门厅）；190＝由独户住宅改成的双户住宅。
- **MSZoning**：土地分区。A＝农业；C＝商业；FV＝浮村住宅区；I＝工业；RH＝高密度住宅区；RL＝低密度住宅区；RP＝低密度住宅园区；RM＝中密度住宅区。
- **LotFrontage**：地块临街边长，单位英尺。
- **LotArea**：地块面积，单位平方英尺。
- **Street**：道路路面。Grvl＝碎石；Pave＝铺装路面。
- **Alley**：后巷路面。Grvl＝碎石；Pave＝铺装路面；NA＝无后巷通道。
- **LotShape**：地块形状。Reg＝规则；IR1＝轻微不规则；IR2＝中度不规则；IR3＝不规则。
- **LandContour**：地势平坦程度。Lvl＝接近平坦；Bnk＝从街道到建筑物处明显上升；HLS＝侧向坡度明显的山坡；Low＝低洼地。
- **Utilities**：公用设施。AllPub＝水、电、燃气和公共排污齐全；NoSewr＝电、燃气和水，使用化粪池；NoSeWa＝只有电和燃气；ELO＝只有电。
- **LotConfig**：地块位置／临街布局。Inside＝内部地块；Corner＝拐角地块；CulDSac＝死胡同尽头；FR2＝两侧临街；FR3＝三侧临街。
- **LandSlope**：地块坡度。Gtl＝缓坡；Mod＝中坡；Sev＝陡坡。
- **Neighborhood**：艾姆斯市内的社区位置。Blmngtn＝Bloomington Heights；Blueste＝Bluestem；BrDale＝Briardale；BrkSide＝Brookside；ClearCr＝Clear Creek；CollgCr＝College Creek；Crawfor＝Crawford；Edwards＝Edwards；Gilbert＝Gilbert；IDOTRR＝爱荷华州交通局／铁路附近；MeadowV＝Meadow Village；Mitchel＝Mitchell；Names＝North Ames；NoRidge＝Northridge；NPkVill＝Northpark Villa；NridgHt＝Northridge Heights；NWAmes＝Northwest Ames；OldTown＝Old Town；SWISU＝爱荷华州立大学南部与西部；Sawyer＝Sawyer；SawyerW＝Sawyer West；Somerst＝Somerset；StoneBr＝Stone Brook；Timber＝Timberland；Veenker＝Veenker。地名保留英文专名。
- **Condition1**：附近的主要外部条件。Artery＝紧邻主干道；Feedr＝紧邻支路；Norm＝一般；RRNn＝距南北向铁路 200 英尺以内；RRAn＝紧邻南北向铁路；PosN＝靠近公园、绿地等正面设施；PosA＝紧邻正面设施；RRNe＝距东西向铁路 200 英尺以内；RRAe＝紧邻东西向铁路。
- **Condition2**：若有多个外部条件，记录第二项；代码含义与 Condition1 相同。
- **BldgType**：住宅类型。1Fam＝独栋独户；2FmCon＝由独户住宅改建的双户住宅；Duplx＝双户住宅；TwnhsE＝端头联排；TwnhsI＝内部联排。
- **HouseStyle**：住宅层数与结构。1Story＝一层；1.5Fin＝一层半且二楼完工；1.5Unf＝一层半且二楼未完工；2Story＝两层；2.5Fin＝两层半且顶层完工；2.5Unf＝两层半且顶层未完工；SFoyer＝分层门厅；SLvl＝错层。

## 房屋质量、建筑与外观

- **OverallQual**：整体用料和装修质量，1–10 分：10＝极佳；9＝优秀；8＝很好；7＝好；6＝高于平均；5＝平均；4＝低于平均；3＝尚可；2＝差；1＝很差。
- **OverallCond**：房屋总体状况，采用与 OverallQual 相同的 1–10 分等级。
- **YearBuilt**：最初建造年份。
- **YearRemodAdd**：改造或加建年份；如果未改造，则与建造年份相同。
- **RoofStyle**：屋顶形状。Flat＝平顶；Gable＝人字顶；Gambrel＝谷仓式双坡顶；Hip＝四面坡顶；Mansard＝法式折线顶；Shed＝单坡顶。
- **RoofMatl**：屋顶材料。ClyTile＝黏土瓦／瓦片；CompShg＝复合屋瓦；Membran＝防水卷材；Metal＝金属；Roll＝卷材；Tar&Grv＝沥青碎石；WdShake＝木劈瓦；WdShngl＝木瓦。
- **Exterior1st**：主要外墙饰面。AsbShng＝石棉瓦；AsphShn＝沥青瓦；BrkComm＝普通砖；BrkFace＝饰面砖；CBlock＝煤渣砖；CemntBd＝水泥板；HdBoard＝硬质板；ImStucc＝仿灰泥；MetalSd＝金属挂板；Other＝其他；Plywood＝胶合板；PreCast＝预制板；Stone＝石材；Stucco＝灰泥；VinylSd＝乙烯基挂板；Wd Sdng＝木挂板；WdShing＝木瓦。
- **Exterior2nd**：次要外墙饰面；类别代码与 Exterior1st 相同，用于房屋有多种外墙材料时。
- **MasVnrType**：砌体贴面类型。BrkCmn＝普通砖；BrkFace＝饰面砖；CBlock＝煤渣砖；None＝无；Stone＝石材。
- **MasVnrArea**：砌体贴面面积，平方英尺。
- **ExterQual**：外墙材料质量。Ex＝极佳；Gd＝好；TA＝普通／典型；Fa＝尚可；Po＝差。
- **ExterCond**：外墙目前状况；等级代码与 ExterQual 相同。
- **Foundation**：地基类型。BrkTil＝砖与瓦；CBlock＝煤渣砖；PConc＝浇筑混凝土；Slab＝混凝土板；Stone＝石材；Wood＝木材。

## 地下室、设备与室内面积

- **BsmtQual**：地下室高度。Ex＝极佳，100 英寸及以上；Gd＝好，90–99 英寸；TA＝普通，80–89 英寸；Fa＝尚可，70–79 英寸；Po＝差，低于 70 英寸；NA＝无地下室。
- **BsmtCond**：地下室总体状况。Ex＝极佳；Gd＝好；TA＝普通，允许轻微潮湿；Fa＝尚可，可能潮湿、开裂或沉降；Po＝差，严重开裂、沉降或积水；NA＝无地下室。
- **BsmtExposure**：地下室是否有向外敞开的墙面／采光条件。Gd＝好；Av＝一般；Mn＝较少；No＝没有；NA＝无地下室。
- **BsmtFinType1**：地下室第一类完工区域的等级。GLQ＝高品质居住区；ALQ＝普通居住区；BLQ＝低于平均的居住区；Rec＝普通娱乐室；LwQ＝低品质；Unf＝未完工；NA＝无地下室。
- **BsmtFinSF1**：地下室第一类完工面积，平方英尺。
- **BsmtFinType2**：地下室第二类完工区域的等级，代码与 BsmtFinType1 相同。
- **BsmtFinSF2**：地下室第二类完工面积，平方英尺。
- **BsmtUnfSF**：地下室未完工面积，平方英尺。
- **TotalBsmtSF**：地下室总面积，平方英尺。
- **Heating**：供暖类型。Floor＝地板炉；GasA＝燃气强制热风炉；GasW＝燃气热水／蒸汽供暖；Grav＝重力式暖炉；OthW＝非燃气热水／蒸汽供暖；Wall＝壁炉式供暖设备。
- **HeatingQC**：供暖质量与状况。Ex＝极佳；Gd＝好；TA＝普通；Fa＝尚可；Po＝差。
- **CentralAir**：是否有中央空调。N＝否；Y＝是。
- **Electrical**：电气系统。SBrkr＝标准断路器和 Romex 电线；FuseA＝超过 60 安培、使用 Romex 电线的保险丝盒；FuseF＝60 安培、主要使用 Romex 电线的保险丝盒；FuseP＝60 安培、主要使用老式旋钮瓷柱布线的保险丝盒；Mix＝混合系统。
- **1stFlrSF**：一楼面积，平方英尺。
- **2ndFlrSF**：二楼面积，平方英尺。
- **LowQualFinSF**：各楼层低品质完工面积，平方英尺。
- **GrLivArea**：地上居住面积，平方英尺。

## 卫浴、房间与功能

- **BsmtFullBath**：地下室完整卫生间数量。
- **BsmtHalfBath**：地下室半卫数量。
- **FullBath**：地上完整卫生间数量。
- **HalfBath**：地上半卫数量。
- **Bedroom**：地上卧室数量，不含地下室卧室。
- **Kitchen**：地上厨房数量。
- **KitchenQual**：厨房质量。Ex＝极佳；Gd＝好；TA＝普通／典型；Fa＝尚可；Po＝差。
- **TotRmsAbvGrd**：地上房间总数，不含卫生间。
- **Functional**：住宅功能状态，除非另有缺陷，默认正常。Typ＝正常；Min1／Min2＝轻微功能缺陷；Mod＝中度缺陷；Maj1／Maj2＝重大缺陷；Sev＝严重损坏；Sal＝仅有残值。
- **Fireplaces**：壁炉数量。
- **FireplaceQu**：壁炉质量。Ex＝极佳的砌筑壁炉；Gd＝位于主居住层的优质砌筑壁炉；TA＝主居住区的普通预制壁炉或地下室砌筑壁炉；Fa＝地下室预制壁炉；Po＝质量较差的独立式炉具；NA＝无壁炉。

## 车库、户外设施与销售

- **GarageType**：车库位置。2Types＝有两种以上车库；Attchd＝与房屋相连；Basment＝地下室车库；BuiltIn＝建在房屋内部，上方通常有房间；CarPort＝车棚；Detchd＝独立车库；NA＝无车库。
- **GarageYrBlt**：车库建造年份。
- **GarageFinish**：车库内部装修状态。Fin＝完工；RFn＝粗装修；Unf＝未完工；NA＝无车库。
- **GarageCars**：按可停车辆数衡量的车库容量。
- **GarageArea**：车库面积，平方英尺。
- **GarageQual**：车库质量。Ex＝极佳；Gd＝好；TA＝普通；Fa＝尚可；Po＝差；NA＝无车库。
- **GarageCond**：车库目前状况；代码与 GarageQual 相同。
- **PavedDrive**：车道是否铺装。Y＝已铺装；P＝部分铺装；N＝泥土／碎石。
- **WoodDeckSF**：木平台面积，平方英尺。
- **OpenPorchSF**：开放式门廊面积，平方英尺。
- **EnclosedPorch**：封闭式门廊面积，平方英尺。
- **3SsnPorch**：三季门廊面积，平方英尺。
- **ScreenPorch**：纱窗门廊面积，平方英尺。
- **PoolArea**：游泳池面积，平方英尺。
- **PoolQC**：游泳池质量。Ex＝极佳；Gd＝好；TA＝普通；Fa＝尚可；NA＝无游泳池。
- **Fence**：围栏质量。GdPrv＝高度隐私围栏；MnPrv＝基本隐私围栏；GdWo＝优质木围栏；MnWw＝普通木／铁丝围栏；NA＝无围栏。
- **MiscFeature**：其他设施。Elev＝电梯；Gar2＝第二车库；Othr＝其他；Shed＝超过 100 平方英尺的棚屋；TenC＝网球场；NA＝无。
- **MiscVal**：其他设施的估计价值，美元。
- **MoSold**：成交月份，1–12。
- **YrSold**：成交年份。
- **SaleType**：交易类型。WD＝普通保证产权契约；CWD＝现金交易的保证产权契约；VWD＝退伍军人贷款交易的保证产权契约；New＝新建住房首次出售；COD＝法院／遗产处置；Con＝常规合同，首付 15%；ConLw＝低首付、低利率合同；ConLI＝低利率合同；ConLD＝低首付合同；Oth＝其他。
- **SaleCondition**：出售条件。Normal＝正常出售；Abnorml＝异常出售，如以物易物、止赎或短售；AdjLand＝连同相邻土地购买；Alloca＝关联房产分别过户，例如公寓与车库；Family＝家庭成员之间交易；Partial＝最后评估时房屋尚未完工，常见于新房。

原文来源：Kaggle *House Prices: Advanced Regression Techniques* 数据文件 `data_description.txt`。这份中文说明仅供理解字段；建模时仍以原始代码和数据文件为准。

