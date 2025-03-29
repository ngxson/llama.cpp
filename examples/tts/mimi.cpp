#include "common.h"
#include "mimi-model.h"

#include <vector>
#include <fstream>


int main(int argc, const char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf codes.txt [output.wav]\n", argv[0]);
        fprintf(stderr, "  Format of codes.txt file: one code per line\n");
        fprintf(stderr, "  Replace codes.txt with dummy0 and dummy1 for testing\n");
        fprintf(stderr, "    dummy0: using code 1, 2, 3,..., 96, used for logits matching\n");
        fprintf(stderr, "    dummy1: using code that will outputs 'wah hello there' sound\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * codes_path = argv[2];
    const char * out_path   = argc < 4 ? "output.wav" : argv[3];

    // load codes
    std::vector<int> codes;
    if (strcmp(codes_path, "dummy0") == 0) {
        printf("Using dummy0 codes\n");
        codes.resize(32 * 3); // [n_codes = 3, n_codes_per_embd = 32]
        int n = 0;
        for (int c = 0; c < 32; c++) {
            for (int r = 0; r < 3; r++) {
                codes[r*32 + c] = n++;
            }
        }
    } else if (strcmp(codes_path, "dummy1") == 0) {
        printf("Using dummy1 codes\n");
        codes = {
            1263 ,1597 ,1596 ,1477 ,1540 ,1720 ,1433 ,118  ,1066 ,1968 ,1096 ,232  ,418  ,566  ,1653 ,2010 ,
            1029 ,1874 ,77   ,1803 ,123  ,908  ,97   ,1616 ,595  ,1170 ,1654 ,1211 ,1967 ,1579 ,1846 ,1462 ,
            1962 ,175  ,1539 ,742  ,1065 ,1226 ,19   ,955  ,528  ,1031 ,659  ,1687 ,1173 ,1802 ,1031 ,1714 ,
            1986 ,582  ,367  ,112  ,1245 ,1386 ,759  ,532  ,1472 ,1790 ,802  ,1213 ,1543 ,1916 ,1251 ,309  ,
            1962 ,1280 ,1943 ,878  ,1588 ,1989 ,568  ,1463 ,1814 ,1095 ,103  ,583  ,976  ,998  ,871  ,587  ,
            247  ,1698 ,1817 ,1024 ,268  ,597  ,45   ,1608 ,1880 ,2047 ,759  ,1578 ,1612 ,49   ,1031 ,1076 ,
            927  ,1202 ,1601 ,1719 ,1670 ,412  ,568  ,1838 ,341  ,1265 ,1279 ,830  ,1997 ,32   ,1369 ,1686 ,
            1307 ,419  ,1143 ,324  ,325  ,572  ,1597 ,1920 ,795  ,915  ,610  ,2000 ,819  ,718  ,1235 ,282  ,
            1912 ,1911 ,141  ,1069 ,1485 ,642  ,1370 ,732  ,284  ,1407 ,1591 ,1002 ,939  ,671  ,951  ,1411 ,
            1887 ,460  ,1588 ,1636 ,1312 ,232  ,969  ,1513 ,1336 ,1185 ,1660 ,4    ,926  ,1243 ,1077 ,1379 ,
            704  ,85   ,257  ,1302 ,1029 ,1717 ,899  ,1345 ,355  ,1915 ,1007 ,315  ,1283 ,779  ,415  ,335  ,
            1848 ,1786 ,469  ,295  ,380  ,1736 ,393  ,765  ,1921 ,836  ,374  ,1649 ,52   ,1633 ,759  ,548  ,
            1922 ,47   ,564  ,893  ,34   ,131  ,1063 ,1657 ,474  ,1960 ,1255 ,1275 ,92   ,976  ,1217 ,483  ,
            105  ,1746 ,1158 ,1557 ,1001 ,512  ,1668 ,1255 ,1045 ,1596 ,613  ,1272 ,1366 ,1147 ,411  ,831  ,
            349  ,692  ,1435 ,2005 ,1465 ,37   ,892  ,95   ,460  ,557  ,1315 ,259  ,1978 ,1838 ,1232 ,2003 ,
            1197 ,111  ,1953 ,1297 ,1843 ,671  ,1687 ,91   ,1788 ,1138 ,1896 ,399  ,615  ,758  ,1423 ,365  ,
            288  ,632  ,876  ,875  ,1156 ,345  ,1189 ,638  ,1527 ,1981 ,1925 ,333  ,1353 ,473  ,1913 ,1443 ,
            1634 ,1373 ,803  ,420  ,192  ,1440 ,1593 ,1925 ,784  ,831  ,552  ,807  ,1942 ,1289 ,612  ,511  ,
            968  ,1091 ,30   ,828  ,1611 ,1241 ,1985 ,596  ,273  ,529  ,1182 ,302  ,726  ,1942 ,733  ,1590 ,
            1564 ,214  ,1156 ,1722 ,1215 ,1837 ,1729 ,1823 ,672  ,116  ,340  ,396  ,721  ,462  ,1615 ,1380 ,
            1459 ,1553 ,636  ,586  ,1148 ,1147 ,1941 ,471  ,876  ,127  ,1938 ,2002 ,1563 ,1121 ,857  ,1179 ,
            1983 ,1324 ,1726 ,1445 ,295  ,270  ,896  ,1947 ,1740 ,1211 ,128  ,1266 ,734  ,715  ,1562 ,285  ,
            1139 ,304  ,526  ,653  ,1270 ,320  ,484  ,22   ,687  ,1065 ,489  ,827  ,993  ,1654 ,431  ,1552 ,
            1418 ,1604 ,455  ,841  ,412  ,848  ,475  ,540  ,1903 ,575  ,584  ,300  ,1079 ,189  ,1481 ,893  ,
            228  ,1577 ,429  ,635  ,106  ,1536 ,176  ,348  ,1733 ,1570 ,537  ,1840 ,798  ,410  ,1714 ,1318 ,
            487  ,332  ,1109 ,1744 ,283  ,692  ,681  ,1744 ,1008 ,1715 ,1956 ,1066 ,1768 ,1645 ,139  ,1967 ,
            897  ,132  ,1010 ,1932 ,277  ,1536 ,1541 ,952  ,19   ,88   ,1663 ,1232 ,1681 ,1878 ,1241 ,1805 ,
            89   ,1401 ,544  ,1061 ,1166 ,267  ,1351 ,1998 ,1623 ,1898 ,425  ,1320 ,2006 ,865  ,1981 ,823  ,
            1243 ,471  ,485  ,1765 ,391  ,1281 ,1607 ,1418 ,116  ,1702 ,1725 ,512  ,1088 ,1375 ,1994 ,1738 ,
            725  ,1471 ,811  ,1251 ,1156 ,1664 ,898  ,1511 ,1872 ,1717 ,444  ,1005 ,254  ,103  ,202  ,1769 ,
            1511 ,433  ,284  ,721  ,1741 ,56   ,615  ,916  ,887  ,1253 ,916  ,535  ,1666 ,1713 ,741  ,873  ,
            447  ,492  ,388  ,321  ,1860 ,1456 ,1658 ,1682 ,848  ,462  ,2034 ,1368 ,1609 ,1887 ,510  ,1516 ,
        };
    } else {
        std::ifstream fin(codes_path);
        if (!fin) {
            fprintf(stderr, "Error: cannot open codes file: %s\n", codes_path);
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            // Skip empty lines
            if (line.empty()) continue;
            try {
                int code = std::stoi(line);
                codes.push_back(code);
            } catch (const std::exception& e) {
                fprintf(stderr, "Error parsing code: %s\n", line.c_str());
                return 1;
            }
        }
        if (codes.empty()) {
            fprintf(stderr, "Error: no codes found in file: %s\n", codes_path);
            return 1;
        }

        printf("Loaded %d codes from %s\n", (int)codes.size(), codes_path);
    }

    mimi_model model(model_path, true);
    std::vector<float> wav_data = model.decode(codes);

    // print first 20 values
    printf("Number of output samples: %d\n", (int)wav_data.size());
    printf("First 20 samples:\n");
    for (int i = 0; i < 20; i++) {
        printf("%2.4f, ", wav_data[i]);
    }
    printf("...\n");

    // write to wav
    printf("Writing to %s\n", out_path);
    save_wav16(out_path, wav_data, model.get_sample_rate());
}
