# DevRec：Accurate developer recommendation for bug resolution

------

该项目实现了一种基于多标签机器学习和LDA主题概率模型的软件缺陷报告分派方法。

代码纯粹本人手写，初次使用python和机器学习在严肃话题，代码注释不够或者混乱还请见谅。

------

代码基本复现了DevRec报告分派方法，但仍缺少降噪处理，在小数据集验证下，性能比原文差一些。
根据原文的验证方法，将数据集11等分，训练第一个数据集，并用地二个测试，接下来训练第一第二个数据集，用第三个预测，以此类推。

    + [OK] read file!
    + [OK] the file has be Standardized!
    + [OK] dictionary is saved in dictionary !
    num_topics = 21

    + [OK] LDA model is saved!
    + [OK] LDA model is ready!

    test only br-analysis
    r5: 1.0
    r10: 1.0

    r5: 1.0
    r10: 1.0

    r5: 0.782051282051282
    r10: 1.0

    r5: 0.7458333333333333
    r10: 1.0

    r5: 0.7125000000000001
    r10: 1.0

    r5: 0.8729166666666668
    r10: 1.0

    r5: 0.8159722222222223
    r10: 1.0

    r5: 0.811437908496732
    r10: 0.9683006535947711

    r5: 0.6088050314465409
    r10: 0.7556603773584903

    r5: 0.6372549019607843
    r10: 1.0

     + [OK]find gama:
     [0.7238926662173903, 0.522715297056122, 0.02707073571840457, 0.23472140478470704, 0.27757847785072043]
     [0.47339674445300617, 0.9199228917603774, 0.19994059851075652, 0.3000426906721848, 0.22600152836999565]

     + [OK] test --- !
     r5: 0
     r10: 0

     r5: 1.0
     r10: 1.0

     r5: 1.0
     r10: 1.0

     r5: 1.0
     r10: 1.0

     r5: 1.0
     r10: 1.0

     r5: 1.0
     r10: 1.0

     r5: 0.2465277777777778
     r10: 0.19791666666666666

     r5: 0.24542483660130718
     r10: 0.30163398692810456

     r5: 0.40220125786163524
     r10: 0.5009433962264151

     r5: 0.6029411764705882
     r10: 0.8725490196078431

## 数据与原论文 Java 包

数据已作为公开数据集 [DevRec: Developer Recommendation for Bug Triage](https://www.kaggle.com/datasets/takiwang/devrec) 发布（CC BY 4.0）。代码（本仓库的 Python 复现和原论文 Java 源码）留在本仓库，不放在 Kaggle 上。

https://www.kaggle.com/datasets/takiwang/devrec

运行 `source/DevRec.py` 前，请从该数据集下载 `aspectj/bugs.csv`，放到 `source/AspectJ.csv`（与脚本中的相对路径一致；该路径已写入 `.gitignore`）。

五项目预处理表在数据集里按 `{gcc,openoffice,mozilla,netbeans,eclipse}/fold{0-9}/` 存放（`reports.parquet`、`terms.parquet`、`vocab.csv`）。原始 ARFF/XML 在 `raw/arff/`。原论文 Java 源码在本仓库 `src/`。本仓库不再包含原来的 `DevRec-java.7z` 或 `source/AspectJ.csv`。
