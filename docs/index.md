# Narrative Operator NLP

NLP 算子 — NarrativeOS 生态的协议优先（Protocol-First）NLP 服务。以 [HanLP](https://github.com/hankcs/HanLP) 为底层引擎，通过 **NSP（Narrative Schema Protocol）** 统一输出，对外提供 MCP / gRPC / FastAPI 三层协议接口。

## 项目文档

```{toctree}
:maxdepth: 1
:caption: 项目文档

protocol
implementation
api_usage
install
deployment
configure
nlp_development_guidelines
gap_analysis
new_word_discovery
classical_chinese
dictionary_architecture
cbdb_integration
contributing
```

## 专题文档

```{toctree}
:maxdepth: 1
:caption: 专题

data_format
tutorial
```

## HanLP 引擎文档（上游参考）

```{toctree}
:caption: HanLP Engine API
:maxdepth: 2

api/hanlp/index
api/common/index
api/trie/index
```

## References

```{toctree}
:caption: References
:maxdepth: 2

references
```


## Acknowledgements

本项目基于 [HanLP](https://github.com/hankcs/HanLP)（Apache 2.0）构建，HanLPv2.1 深受 [AllenNLP](https://allennlp.org/) 与 [SuPar](https://pypi.org/project/supar/) 的启发。

