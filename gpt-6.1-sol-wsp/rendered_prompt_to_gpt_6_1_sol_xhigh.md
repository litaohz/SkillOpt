We are testing Weight-Space Programming on Qwen/Qwen3.5-4B (32 layers, hybrid Linear-Attention + Full-Attention at layers 3,7,11,15,19,23,27,31; LoRA r=16, alpha=32) for Browser Tab Grouping.
We have 4 adapters starting from `sft2000`:
- `sft2000`: v1 93.85% P / 68.68% R; v2.1 90.14% P / 48.00% R (giant-bucket over-merging).
- `opsd_astra` (150 steps on-policy self-distillation with Astra reference): v1 97.99% P / 70.20% R; v2.1 92.36% P / 54.60% R.
- `sft_astra` (150 steps standard SFT on same 1,200 Astra answers): v1 97.13% P / 60.30% R (-9.90pp Recall drop!); v2.1 94.21% P / 57.86% R (+3.26pp higher Recall).
In the 1,200 Astra training labels, 13.03% of group names contain `' & '` (vs 1.43% in original SFT) and Ungrouped tabs are 15.3% higher.

Weight-space SVD & vocabulary projection table (`d_OPSD = W_opsd - W_0`, `d_SFT = W_sft - W_0`):
Layer | Module | cos(d_OPSD,d_SFT) | ||d_OPSD|| | ||d_SFT|| | top1_ratio(OPSD/SFT) | vocab_proj(& OPSD/SFT) | u1_SFT_top_pos | u1_OPSD_top_pos
L00 | mlp.down_proj        | 0.1693 | 0.0468 | 0.0535 | 0.168/0.168 | 0.0007/0.0014 | [' oxid', 'undy', 'nios'] | [' defi', 'entr', ' Ha']
L00 | linear_attn.out_proj | 0.1633 | 0.0435 | 0.0519 | 0.375/0.455 | 0.0006/0.0008 | [' mirrored', '...\\', 'san'] | ['拟', ' olev', ' os']
L00 | mlp.gate_proj        | 0.1528 | 0.0744 | 0.0843 | 0.164/0.155 | - | - | -
L01 | mlp.down_proj        | 0.1543 | 0.0468 | 0.0533 | 0.192/0.192 | 0.0009/0.0009 | ['haw', 'рать', 'TK'] | ['ram', 'auge', 'ingham']
L01 | linear_attn.out_proj | 0.1459 | 0.0407 | 0.0454 | 0.210/0.189 | 0.0005/0.0010 | ['одно', '汾', '供图'] | ['doc', '展', 'ikas']
L01 | mlp.gate_proj        | 0.1384 | 0.0732 | 0.0818 | 0.202/0.181 | - | - | -
L02 | mlp.down_proj        | 0.1873 | 0.0502 | 0.0603 | 0.284/0.298 | 0.0006/0.0019 | [' mal', 'öv', 'fav'] | [' Gem', 'ịa', 'dest']
L02 | linear_attn.out_proj | 0.1713 | 0.0401 | 0.0462 | 0.148/0.159 | 0.0009/0.0010 | ['?f', '령', ' tille'] | ['又将', 'schau', '堪']
L02 | mlp.gate_proj        | 0.1585 | 0.0741 | 0.0841 | 0.186/0.167 | - | - | -
L03 | mlp.down_proj        | 0.2039 | 0.0475 | 0.0641 | 0.195/0.274 | 0.0011/0.0012 | ['gv', 'Ui', 'حي'] | ['amble', 'ablemente', 'тім']
L03 | self_attn.o_proj     | 0.1899 | 0.0420 | 0.0516 | 0.327/0.305 | 0.0006/0.0009 | ['ingly', 'ive', 'ikel'] | ['cit', '不可思议', ' разбира']
L03 | mlp.gate_proj        | 0.1632 | 0.0759 | 0.0937 | 0.275/0.316 | - | - | -
L04 | mlp.down_proj        | 0.2134 | 0.0470 | 0.0650 | 0.191/0.299 | 0.0009/0.0017 | [' accompagne', 'acre', '�'] | ['titles', 'vision', 'شهد']
L04 | linear_attn.out_proj | 0.1855 | 0.0408 | 0.0531 | 0.200/0.274 | 0.0008/0.0016 | ['aurus', '((', '撒'] | [' manifest', 'oda', ' fol']
L04 | mlp.gate_proj        | 0.1786 | 0.0793 | 0.0969 | 0.364/0.375 | - | - | -
L05 | mlp.down_proj        | 0.2482 | 0.0491 | 0.0703 | 0.239/0.316 | 0.0007/0.0010 | ['сихо', 'ady', ' itself'] | ['û', '明', 'aq']
L05 | linear_attn.out_proj | 0.2213 | 0.0417 | 0.0567 | 0.251/0.251 | 0.0006/0.0011 | ['六', 'rita', '成为'] | ['atch', 'atches', 'edia']
L05 | mlp.gate_proj        | 0.2021 | 0.0769 | 0.0992 | 0.235/0.271 | - | - | -
L06 | mlp.down_proj        | 0.2819 | 0.0497 | 0.0693 | 0.251/0.248 | 0.0008/0.0014 | ['pong', 'ificare', '极了'] | ['heat', ' بيد', '-educated']
L06 | linear_attn.out_proj | 0.2634 | 0.0423 | 0.0583 | 0.310/0.305 | 0.0010/0.0025 | ['zie', ' kir', '追'] | ['�', '-cent', '료가']
L06 | mlp.gate_proj        | 0.2413 | 0.0792 | 0.1041 | 0.402/0.433 | - | - | -
L07 | mlp.down_proj        | 0.2584 | 0.0524 | 0.0816 | 0.321/0.337 | 0.0009/0.0022 | ['高等', 'inho', '陷阱'] | ['體系', 'ৈ', '期']
L07 | self_attn.o_proj     | 0.2010 | 0.0444 | 0.0737 | 0.489/0.509 | 0.0005/0.0017 | ['格式', 'یش', 'obo'] | ['udge', 'illery', '香氛']
L07 | mlp.gate_proj        | 0.2538 | 0.0815 | 0.1072 | 0.484/0.476 | - | - | -
L08 | mlp.down_proj        | 0.2818 | 0.0486 | 0.0767 | 0.205/0.297 | 0.0009/0.0026 | ['ียด', 'zor', 'akil'] | [' partial', '\n\n', 'ASM']
L08 | linear_attn.out_proj | 0.2738 | 0.0417 | 0.0651 | 0.206/0.228 | 0.0009/0.0023 | ['eral', 'uster', '道理'] | ['超前', 'pla', '对象']
L08 | mlp.gate_proj        | 0.2789 | 0.0822 | 0.1130 | 0.506/0.502 | - | - | -
L09 | mlp.down_proj        | 0.3125 | 0.0504 | 0.0795 | 0.230/0.210 | 0.0008/0.0015 | ['际', '了自己', 'izr'] | ['vid', 'gh', '援']
L09 | linear_attn.out_proj | 0.3244 | 0.0434 | 0.0660 | 0.292/0.337 | 0.0010/0.0018 | ['accia', '不久的', ' pronto'] | ['ทั้งสิ้น', 'صر', '-lang']
L09 | mlp.gate_proj        | 0.2953 | 0.0801 | 0.1106 | 0.426/0.412 | - | - | -
L10 | mlp.down_proj        | 0.3056 | 0.0497 | 0.0752 | 0.258/0.237 | 0.0009/0.0016 | ['壮', '狩', 'asting'] | [' pat', 'idon', 'ESC']
L10 | linear_attn.out_proj | 0.3445 | 0.0440 | 0.0765 | 0.223/0.365 | 0.0011/0.0024 | ['apper', '符', '状'] | ['充分', ' Stout', '那边']
L10 | mlp.gate_proj        | 0.3080 | 0.0868 | 0.1241 | 0.540/0.505 | - | - | -
L11 | mlp.down_proj        | 0.3487 | 0.0500 | 0.0910 | 0.327/0.491 | 0.0010/0.0014 | ['ertime', 'ress', 'IMS'] | ['etat', 'aju', 'ais']
L11 | self_attn.o_proj     | 0.2999 | 0.0425 | 0.0750 | 0.212/0.350 | 0.0008/0.0014 | ['allet', '期中', 'uster'] | ['�', '这个问题的', '相关问题']
L11 | mlp.gate_proj        | 0.3451 | 0.0827 | 0.1303 | 0.469/0.385 | - | - | -
L12 | mlp.down_proj        | 0.3294 | 0.0532 | 0.0977 | 0.387/0.411 | 0.0014/0.0021 | ['令第', '霖', 'ivat'] | ['以身', '及个人', 'otes']
L12 | linear_attn.out_proj | 0.2907 | 0.0434 | 0.0720 | 0.165/0.278 | 0.0010/0.0018 | [' séqu', ' boils', '蔻'] | ['apo', ' apo', 'apa']
L12 | mlp.gate_proj        | 0.3300 | 0.0832 | 0.1249 | 0.407/0.437 | - | - | -
L13 | mlp.down_proj        | 0.2763 | 0.0597 | 0.0914 | 0.463/0.400 | 0.0011/0.0015 | ['aget', '进行操作', 'vert'] | ['adolu', 'apa', 'iat']
L13 | linear_attn.out_proj | 0.2858 | 0.0436 | 0.0688 | 0.165/0.233 | 0.0009/0.0016 | ['odial', ' ditar', '看'] | ['合理化', '遍', '辅']
L13 | mlp.gate_proj        | 0.3616 | 0.0907 | 0.1377 | 0.551/0.591 | - | - | -
L14 | mlp.down_proj        | 0.2391 | 0.0583 | 0.0878 | 0.501/0.406 | 0.0008/0.0024 | ['ohl', ' сход', 'empl'] | [' întâl', 'itamente', 'mbH']
L14 | linear_attn.out_proj | 0.2764 | 0.0450 | 0.0694 | 0.220/0.252 | 0.0010/0.0011 | ['不如意', ' principals', 'Principal'] | ['go', 'ho', '妾']
L14 | mlp.gate_proj        | 0.3224 | 0.0875 | 0.1332 | 0.379/0.428 | - | - | -
L15 | mlp.down_proj        | 0.2796 | 0.0547 | 0.0939 | 0.280/0.387 | 0.0008/0.0033 | ['在使用过程中', '深受广大', '基地'] | ['韬', 'itr', 'ysz']
L15 | self_attn.o_proj     | 0.2274 | 0.0441 | 0.0822 | 0.256/0.561 | 0.0008/0.0025 | ['eon', 'allet', '在使用过程中'] | ['五金交电', 'uri', 'aru']
L15 | mlp.gate_proj        | 0.3001 | 0.0887 | 0.1474 | 0.561/0.540 | - | - | -
L16 | mlp.down_proj        | 0.1578 | 0.0543 | 0.0956 | 0.409/0.334 | 0.0010/0.0041 | ['効', 'izion', '适'] | ['的安排', '乍', 'oola']
L16 | linear_attn.out_proj | 0.2199 | 0.0450 | 0.0770 | 0.228/0.335 | 0.0010/0.0033 | ['fest', 'izion', 'setUp'] | ['GL', '方面', '方面有']
L16 | mlp.gate_proj        | 0.2983 | 0.0934 | 0.1567 | 0.546/0.493 | - | - | -
L17 | mlp.down_proj        | 0.1749 | 0.0538 | 0.0948 | 0.386/0.382 | 0.0009/0.0039 | ['adv', 'akes', '一聲'] | ['ende', '-,', 'oping']
L17 | linear_attn.out_proj | 0.2554 | 0.0502 | 0.0808 | 0.394/0.515 | 0.0011/0.0041 | ['udou', '按照规定', '不肯'] | ['éia', ' performa', 'ște']
L17 | mlp.gate_proj        | 0.2669 | 0.0865 | 0.1453 | 0.382/0.405 | - | - | -
L18 | mlp.down_proj        | 0.2102 | 0.0525 | 0.0807 | 0.370/0.448 | 0.0014/0.0072 | [' &', '&', '及各'] | ['�', 'lessly', 'te']
L18 | linear_attn.out_proj | 0.2008 | 0.0441 | 0.0711 | 0.218/0.476 | 0.0012/0.0045 | [' &', 'ton', 'olt'] | ['还被', 'accio', '情况的']
L18 | mlp.gate_proj        | 0.2481 | 0.0830 | 0.1209 | 0.357/0.357 | - | - | -
L19 | mlp.down_proj        | 0.0883 | 0.0527 | 0.0866 | 0.338/0.386 | 0.0011/0.0042 | ['oker', 'ей', '电器'] | ['落的', 'Pas', '合同履行']
L19 | self_attn.o_proj     | 0.2068 | 0.0448 | 0.0685 | 0.227/0.470 | 0.0008/0.0052 | ['什么', ' separator', '有关'] | ['gee', ' Gratuit', 'asikan']
L19 | mlp.gate_proj        | 0.2266 | 0.0826 | 0.1220 | 0.475/0.477 | - | - | -
L20 | mlp.down_proj        | 0.2225 | 0.0506 | 0.0812 | 0.312/0.502 | 0.0006/0.0071 | [' &', '&', '学习与'] | ['垂', 'transport', ' tik']
L20 | linear_attn.out_proj | 0.1445 | 0.0429 | 0.0640 | 0.275/0.400 | 0.0008/0.0070 | [' &', '&', '及'] | [' нев', '乌', '令人']
L20 | mlp.gate_proj        | 0.2897 | 0.0881 | 0.1277 | 0.506/0.434 | - | - | -
L21 | mlp.down_proj        | 0.2217 | 0.0527 | 0.0754 | 0.307/0.441 | 0.0009/0.0078 | [' &', '&', ' (&'] | ['عان', '工业协会', ' undermin']
L21 | linear_attn.out_proj | 0.1811 | 0.0429 | 0.0755 | 0.400/0.592 | 0.0005/0.0064 | ['学习与', ' &', '性及'] | ['boljš', 'esub', ' downlo']
L21 | mlp.gate_proj        | 0.2063 | 0.0813 | 0.1160 | 0.506/0.524 | - | - | -
L22 | mlp.down_proj        | 0.1905 | 0.0477 | 0.0704 | 0.297/0.394 | 0.0009/0.0089 | [' &', '&', '学习与'] | ['epad', 'ächer', 'ھر']
L22 | linear_attn.out_proj | 0.1754 | 0.0415 | 0.0617 | 0.248/0.405 | 0.0008/0.0070 | ['lator', 'kers', 'parer'] | ['อยู่น', 'عان', 'かさ']
L22 | mlp.gate_proj        | 0.1137 | 0.0845 | 0.1250 | 0.512/0.406 | - | - | -
L23 | mlp.down_proj        | 0.2082 | 0.0502 | 0.0762 | 0.290/0.401 | 0.0010/0.0111 | [' "**', '斌', 'icorn'] | [' Grosso', '埠', ' Adult']
L23 | self_attn.o_proj     | 0.1685 | 0.0411 | 0.0572 | 0.199/0.380 | 0.0009/0.0059 | ['学习与', ' &', ' management'] | [' di', ' education', 'ready']
L23 | mlp.gate_proj        | 0.0813 | 0.0835 | 0.1229 | 0.557/0.486 | - | - | -
L24 | mlp.down_proj        | 0.1535 | 0.0475 | 0.0699 | 0.269/0.384 | 0.0012/0.0100 | [' &', ' learning', '&'] | ['history', 'History', ' history']
L24 | linear_attn.out_proj | 0.0851 | 0.0407 | 0.0545 | 0.234/0.286 | 0.0006/0.0074 | ['知之', 'StateException', ' lest'] | ['煮沸', 'kowych', 'ography']
L24 | mlp.gate_proj        | 0.1048 | 0.0813 | 0.1162 | 0.514/0.430 | - | - | -
L25 | mlp.down_proj        | 0.0854 | 0.0511 | 0.0731 | 0.363/0.384 | 0.0012/0.0055 | [' learning', ' Learning', 'Learning'] | ['Basket', '谓', 'iction']
L25 | linear_attn.out_proj | 0.1465 | 0.0403 | 0.0598 | 0.225/0.421 | 0.0008/0.0081 | ['เครือ', 'chestra', 'صار'] | [' Promise', '教育的', 'гал']
L25 | mlp.gate_proj        | 0.0938 | 0.0796 | 0.1131 | 0.479/0.410 | - | - | -
L26 | mlp.down_proj        | 0.0517 | 0.0505 | 0.0737 | 0.338/0.295 | 0.0010/0.0075 | ['osoph', 'iph', '-items'] | ['ography', '永和', 'eggi']
L26 | linear_attn.out_proj | 0.1516 | 0.0405 | 0.0587 | 0.199/0.335 | 0.0006/0.0063 | ['azor', '乐团', ' explic'] | ['ders', 'mans', 'лли']
L26 | mlp.gate_proj        | 0.1025 | 0.0832 | 0.1052 | 0.526/0.514 | - | - | -
L27 | mlp.down_proj        | 0.1668 | 0.0491 | 0.0791 | 0.352/0.378 | 0.0016/0.0090 | ['work', 'Education', ' work'] | [' Browser', '恶', 'B']
L27 | self_attn.o_proj     | 0.1585 | 0.0416 | 0.0635 | 0.210/0.306 | 0.0012/0.0087 | ['Grupo', ' Grupo', ' hiburan'] | ['榜', 'dings', '喱']
L27 | mlp.gate_proj        | 0.0714 | 0.1083 | 0.1734 | 0.728/0.670 | - | - | -
L28 | mlp.down_proj        | 0.2738 | 0.0478 | 0.0819 | 0.214/0.384 | 0.0010/0.0094 | ['Port', '题', ' Port'] | ['edy', 'Education', '同']
L28 | linear_attn.out_proj | 0.1302 | 0.0416 | 0.0595 | 0.261/0.204 | 0.0009/0.0067 | [' group', 'grup', 'Group'] | ['的一种', 'iris', 'quelle']
L28 | mlp.gate_proj        | 0.0654 | 0.0970 | 0.1391 | 0.735/0.643 | - | - | -
L29 | mlp.down_proj        | 0.2296 | 0.0494 | 0.0779 | 0.268/0.350 | 0.0011/0.0122 | [' &', '&', ' learning'] | ['Ch', 'ipers', 'ุ่ม']
L29 | linear_attn.out_proj | 0.1282 | 0.0413 | 0.0622 | 0.272/0.249 | 0.0007/0.0083 | ['Education', 'Grupo', ' Grupo'] | ['YouTube', ' YouTube', 'ья']
L29 | mlp.gate_proj        | 0.1322 | 0.0785 | 0.1095 | 0.532/0.459 | - | - | -
L30 | mlp.down_proj        | 0.2146 | 0.0469 | 0.0819 | 0.211/0.355 | 0.0014/0.0110 | [' Learning', ' &', 'Learning'] | ['Business', ' Business', 'Basket']
L30 | linear_attn.out_proj | 0.1546 | 0.0436 | 0.0584 | 0.256/0.270 | 0.0006/0.0086 | [' &', '&', '.&'] | ['Programming', 'programming', ' Programming']
L30 | mlp.gate_proj        | 0.1172 | 0.0756 | 0.1049 | 0.434/0.441 | - | - | -
L31 | mlp.down_proj        | 0.2143 | 0.0395 | 0.0699 | 0.324/0.461 | 0.0046/0.0127 | [':', ' ', '.'] | [' Leverkus', 'angeb', '挖掘公司']
L31 | self_attn.o_proj     | 0.1133 | 0.0415 | 0.0654 | 0.418/0.446 | 0.0021/0.0118 | [' &', ' learning', ' Learning'] | ['History', 'history', ' history']
L31 | mlp.gate_proj        | 0.1846 | 0.0699 | 0.0969 | 0.291/0.372 | - | - | -

1. Explain the exact mechanistic difference across layers (specifically why L08-L15 align up to cos=0.36 while L18-L31 diverge down to cos=0.05 with `sft_astra` top-1 singular vector writing `' &'`, `'学习与'`, `'Learning'` into the residual stream).
2. Propose 3 exact Zero-Training Weight-Space Surgical Edits to combine OPSD's v1 Recall (~70%+) and Astra-SFT's v2.1 splitting (~58%+).
