// ── data.js — static configuration for the demo page ────────────
// Sample texts — extracted 1:1 from the original inline sample cards.
// Each card: text (full sample), lang (language mode), label, preview, dict (optional custom dict).
export const SAMPLES = [
  { text: '阿里巴巴集团成立于1999年，由马云等人在杭州创立。公司最初专注于B2B电子商务，为中小企业提供在线交易平台。2003年，阿里巴巴推出淘宝网，进军C2C电商领域，迅速成为中国最大的网上购物平台。2004年，支付宝成立，解决了网络交易的信任问题，后来发展为全球最大的第三方支付平台之一。2009年，阿里云成立，致力于打造云计算基础设施，现已成为中国最大的公有云服务商。2014年，阿里巴巴在纽约证券交易所上市，成为当时全球最大的IPO。此后，阿里巴巴持续拓展业务版图，涵盖电商、云计算、数字媒体、物流和本地生活等多个领域。2023年，阿里巴巴宣布启动"1+6+N"组织变革，将业务拆分为六大业务集团，以提升各业务单元的灵活性和竞争力。', lang: 'auto', label: '📄 企业+摘要', preview: '阿里巴巴集团成立于1999年，由马云等人在杭州创立。公司最初专注于B2B电子商务……' },
  { text: '张三于2020年从清华大学计算机系毕业，获得博士学位。他的研究方向是自然语言处理和知识图谱。毕业后，他加入百度研究院，担任高级算法工程师。在百度工作期间，他主导开发了新一代中文分词系统，准确率提升了15%。2022年，他参与构建了百度知识图谱2.0，覆盖实体超过10亿。2023年，他转投阿里巴巴达摩院，继续深耕大语言模型方向。目前，他负责阿里通义千问模型的优化工作，在推理速度和生成质量方面取得了显著进展。', lang: 'modern', label: '📄 人物+摘要', preview: '张三于2020年从清华大学计算机系毕业，获得博士学位。他的研究方向是自然语言处理……' },
  { text: '2024年5月28日，百度在北京举办了AI开发者大会。大会上，百度正式发布了文心大模型4.0专业版，支持超长上下文窗口，可处理超过25万字的文档。百度CEO李彦宏表示，文心大模型已在金融、医疗、法律等多个行业落地应用。同日，百度还发布了飞桨深度学习框架6.0版本，新增了对多模态任务的全面支持。在自动驾驶领域，百度Apollo宣布累计测试里程突破1000万公里，已在10个城市开展Robotaxi商业化运营。此外，百度还发布了智能云新战略，提出"云智一体"理念，将AI能力深度融入云计算服务。百度还与清华大学联合成立了AI联合实验室，聚焦大模型基础研究和产业应用。', lang: 'modern', label: '📄 事件+摘要', preview: '2024年5月28日，百度在北京举办了AI开发者大会。大会上，百度正式发布了文心大模型……' },
  { text: '苹果公司在加州库比蒂诺的总部发布了新款iPhone 16 Pro。该产品搭载A18 Pro芯片，采用台积电3纳米工艺，CPU性能提升25%，GPU性能提升40%。iPhone 16 Pro配备6.3英寸和6.9英寸两种尺寸的OLED屏幕，支持120Hz自适应刷新率。在AI方面，苹果推出了Apple Intelligence，深度集成于Siri和系统级应用中。该产品起售价为799美元，于9月20日正式开售。分析师预计，iPhone 16系列首季度销量将超过8000万台，同比增长10%。苹果公司总部位于加州库比蒂诺，是全球市值最高的科技公司之一。', lang: 'modern', label: '📄 产品+摘要', preview: '苹果公司在加州库比蒂诺的总部发布了新款iPhone 16 Pro。该产品搭载A18 Pro芯片……', dict: 'iPhone 16 Pro A18 Pro 库比蒂诺 Apple Intelligence' },
  { text: '北京立方庭位于海淀区中关村核心区域，是一栋甲级写字楼。总建筑面积约5万平方米，地上28层，地下3层。建筑高度120米，于2018年竣工投入使用。立方庭采用全玻璃幕墙设计，获得LEED金级认证。该建筑配备智能楼宇管理系统，可实现能耗自动优化。周边交通便利，距离地铁4号线中关村站仅200米。立方庭入驻企业包括多家知名科技公司和金融机构。物业管理由仲量联行负责，提供24小时安保和前台服务。', lang: 'auto', label: '📄 建筑+摘要', preview: '北京立方庭位于海淀区中关村核心区域，是一栋甲级写字楼。总建筑面积约5万平方米……' },
  { text: '陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕，辍耕之垄上，怅恨久之，曰：\'苟富贵，无相忘。\'佣者笑而应曰：\'若为佣耕，何富贵也？\'陈涉太息曰：\'嗟乎，燕雀安知鸿鹄之志哉！\'二世元年七月，发闾左適戍渔阳，九百人屯大泽乡。陈胜吴广皆次当行，为屯长。会天大雨，道不通，度已失期。失期，法皆斩。陈胜吴广乃谋曰：\'今亡亦死，举大计亦死，等死，死国可乎？\'陈胜曰：\'天下苦秦久矣。\'卒买鱼烹食，得鱼腹中书，曰\'陈胜王\'。卒皆夜惊恐。又间令吴广之次所旁丛祠中，夜篝火，狐鸣呼曰：\'大楚兴，陈胜王。\'卒皆夜惊恐。旦日，卒中往往语，皆指目陈胜。', lang: 'classical', label: '🏯 陈涉世家+摘要', preview: '陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕……' },
  { text: '北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。鹏之背，不知其几千里也。怒而飞，其翼若垂天之云。是鸟也，海运则将徙于南冥。南冥者，天池也。《齐谐》者，志怪者也。《谐》之言曰：\'鹏之徙于南冥也，水击三千里，抟扶摇而上者九万里，绝云气，负青天，然后图南，且适南冥也。\'斥鴳笑之曰：\'彼且奚适也？我腾跃而上，不过数仞而下，翱翔蓬蒿之间，此亦飞之至也。\'而彼且奚适也？此小大之辩也。', lang: 'classical', label: '🏯 逍遥游+摘要', preview: '北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏……' },
  { text: '十年春，齐师伐我。公将战，曹刿请见。其乡人曰：\'肉食者谋之，又何间焉？\'刿曰：\'肉食者鄙，未能远谋。\'乃入见。问：\'何以战？\'公曰：\'衣食所安，弗敢专也，必以分人。\'对曰：\'小惠未遍，民弗从也。\'公曰：\'牺牲玉帛，弗敢加也，必以信。\'对曰：\'小信未孚，神弗福也。\'公曰：\'小大之狱，虽不能察，必以情。\'对曰：\'忠之属也，可以一战。战则请从。\'公与之乘，战于长勺。公将鼓之。刿曰：\'未可。\'齐人三鼓。刿曰：\'可矣。\'齐师败绩。公将驰之。刿曰：\'未可。\'下视其辙，登轼而望之，曰：\'可矣。\'遂逐齐师。既克，公问其故。对曰：\'夫战，勇气也。一鼓作气，再而衰，三而竭。彼竭我盈，故克之。\' ', lang: 'classical', label: '🏯 曹刿论战+摘要', preview: '十年春，齐师伐我。公将战，曹刿请见。其乡人曰：\'肉食者谋之……' },
  { text: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in Cupertino, California. The company initially focused on personal computers, with the Apple I and Apple II becoming commercial successes. In 1984, Apple introduced the Macintosh, the first mass-market computer with a graphical user interface. After Jobs was ousted in 1985, he returned in 1997 and transformed the company. Under his leadership, Apple launched the iMac, iPod, iPhone, and iPad, revolutionizing multiple industries. The iPhone, released in 2007, became the best-selling smartphone worldwide. As of 2024, Apple is the world\'s most valuable company, with a market capitalization exceeding $3 trillion. The company generates over $380 billion in annual revenue, with services including Apple Music, iCloud, and the App Store contributing significantly to its growth.', lang: 'english', label: '🇬🇧 Apple+Summary', preview: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne……' },
  { text: 'Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4, 1975, in Albuquerque, New Mexico. The company\'s first product was an interpreter for the BASIC programming language. In 1981, Microsoft developed MS-DOS for IBM\'s first personal computer, which became the industry standard. Windows 3.0, released in 1990, was the first truly successful graphical operating system. Windows 95 introduced the Start menu and taskbar, becoming a cultural phenomenon. In 2014, Satya Nadella became CEO and shifted Microsoft\'s focus to cloud computing and AI. Azure, launched in 2010, became the second-largest cloud platform globally. Microsoft acquired LinkedIn in 2016 and GitHub in 2018, expanding its developer ecosystem. The company\'s Copilot AI assistant, powered by OpenAI\'s GPT models, has been integrated across its product suite. As of 2024, Microsoft is valued at over $3 trillion.', lang: 'english', label: '🇬🇧 Microsoft+Summary', preview: 'Microsoft Corporation was founded by Bill Gates and Paul Allen……' },
];

// ── Data: tabs & panels ─────────────────────────────────────────────────────
// id must match the panel element id; the first tab/panel is active by default.
// group is used to render section separators in the tab bar.
export const TABS = [
  { id: 'overview',   label: '📋 总览',            group: '基础分析' },
  { id: 'nsp',        label: '📊 NSP 结构化',      group: '基础分析' },
  { id: 'pretty',     label: '🎨 HanLP 原生可视化', group: '基础分析' },
  { id: 'depsvg',     label: '🧬 依存树 SVG',      group: '基础分析' },
  { id: 'discover',   label: '🔍 新词发现',        group: '深度分析' },
  { id: 'patterns',   label: '📊 句式模式',        group: '深度分析' },
  { id: 'coref',      label: '🔗 指代消解',        group: '深度分析' },
  { id: 'langdetect', label: '🏯 语言检测',        group: '深度分析' },
  { id: 'summary',    label: '📝 摘要',            group: '生成' },
  { id: 'title',      label: '🏷️ 标题',            group: '生成' },
  { id: 'json',       label: '{ } JSON Raw',       group: '原始数据' },
  { id: 'api',        label: '📋 API 接口',        group: '原始数据' },
];
