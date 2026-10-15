# Evaluation: ML, məntiq və riyaziyyat

Bu sənəd sənin hazırladığın kodu izah edir. Metrik modeli öyrətmir: hazır proqnozların keyfiyyətini ölçür. Pipeline belədir: training → model → test proqnozu → həqiqi test cavabı ilə müqayisə → metrik.

`y_true`: həqiqi cavablar. `y_pred`: modelin cavabları. Eyni indeks eyni evə aid olmalıdır. `scores`: premium ehtimalı və ya SVM margin-i; böyük qiymət premium deməkdir. Label 0/1-dir, score isə davamlı ədəddir.

## Regression

Hər ev üçün xəta eᵢ = yᵢ − ŷᵢ.

MAE = (1/n) Σ |eᵢ|. Mütləq qiymət mənfi və müsbət xətaların bir-birini ləğv etməsinə mane olur. Qiymətlər AZN-dirsə, MAE də AZN-dir. Kiçik qiymət yaxşıdır.

RMSE = √[(1/n) Σ eᵢ²]. Kvadrat böyük xətalara daha çox ağırlıq verir. Kök nəticəni yenidən qiymətin vahidinə qaytarır. Kiçik qiymət yaxşıdır. RMSE adətən MAE-dən az olmur.

Nümunə: həqiqət [10,20,30], proqnoz [12,18,33]. Xətalar [-2,2,-3]. Mütləq xətalar [2,2,3], MAE=7/3≈2.3333. Kvadratlar [4,4,9], RMSE=√(17/3)≈2.3805.

R² = 1 − SS_res/SS_tot. SS_res=Σ(yᵢ−ŷᵢ)² modelin səhvləridir. SS_tot=Σ(yᵢ−ȳ)² həqiqi qiymətlərin öz ortasından uzaqlığıdır. Bu nümunədə orta 20, SS_tot=200, SS_res=17, R²=0.915. Bu accuracy və ya 'evlərin 91.5%-i doğru tapılıb' demək deyil. Bu nümunədə model, qiymətləndirilən cavabların ortasını hər evə verməklə müqayisədə kvadrat xətanı 91.5% azaldır.

R²=1 mükəmməl nəticədir, 0 orta proqnoz qədər nəticədir, mənfi qiymət ortanı proqnozlaşdırmaqdan pisdir. Bir nümunə ilə R² təyin olunmur: NaN qaytarırıq. Hədəf sabitdirsə düsturda məxrəc 0 olur; praktik konvensiya ilə mükəmməl proqnoza 1, digərinə 0 qaytarırıq.

Log-price modeli üçün hər iki array eyni miqyasda olmalıdır. Qiymət miqyasında ölçmək üçün proqnozun transformasiyası geri çevrilir: log üçün exp, log1p üçün expm1. Log RMSE-ni AZN kimi təqdim etmək olmaz.

## Classification

Premium=1 pozitiv sinifdir. Standard=0 neqativ sinifdir. TP: premium evə premium demək. TN: standard evə standard demək. FP: standard evə səhvən premium demək. FN: premium evə səhvən standard demək.

Confusion matrix = [[TN,FP],[FN,TP]]. Sətirlər həqiqət, sütunlar proqnozdur.

Precision = TP/(TP+FP): premium dediklərimizin nə qədəri həqiqətən premiumdur? Recall = TP/(TP+FN): həqiqi premium evlərin nə qədərini tapmışıq? F1 = 2PR/(P+R): precision və recall-un harmonik ortası. Biri aşağıdırsa F1 də azalır. Hamısında böyük qiymət yaxşıdır. Məxrəc sıfırdırsa burada 0 seçirik; bunu reportda qeyd etmək lazımdır.

Nümunə: həqiqət [0,0,1,1], proqnoz [0,1,0,1]. TN=FP=FN=TP=1; precision=recall=F1=0.5.

ROC əyrisi müxtəlif hədlərdə TPR=recall ilə FPR=FP/(FP+TN) əlaqəsini göstərir. ROC-AUC bu əyrinin altında qalan sahədir. Biz əyrini çəkmədən ekvivalent rank düsturu ilə sahəni hesablayırıq.

AUC-nin mənası: təsadüfi premium-standard cütündə premium-a daha böyük score verilmə ehtimalı; bərabərlik yarım uğur sayılır. 1 ideal sıralama, 0.5 təsadüfi sıralamanın gözlənilən səviyyəsi, 0 tərs sıralamadır. 0.5 bir konkret modelin təsadüfi olduğunu sübut etmir. Testdə yalnız bir sinif varsa AUC təyin olunmur: NaN.

Score-lar [.1,.4,.35,.8], label-lar [0,0,1,1] olduqda premium score-ları [.35,.8], standard [.1,.4]-dür. Dörd cüt müqayisəsindən üçündə premium böyükdür: AUC=3/4=.75.

Rank düsturu: AUC = [R₊ − n₊(n₊+1)/2] / (n₊ n₋). R₊ pozitivlərin rank cəmidir. n₊ pozitiv, n₋ neqativ sayıdır. Pozitivlərin öz aralarındakı rank payını çıxırıq; qalan kəmiyyət neqativlərlə qazanılan müqayisələrdir. Eyni score-lara orta rank verilir: məsələn 2 və 3-cü yerlər eynidirsə hər ikisinin rank-ı 2.5 olur.

## Faylların sətir-sətir izahı

Sətir nömrələri bu sənəd yaradılan andakı kodla eynidir. Boş sətirlər yalnız oxunaqlılıq üçündür. Docstring və şərhlər alqoritm əmri deyil, izahdır.


### src/evaluate.py

**Sətir 1:** `"""NumPy metrics: premium=1, standard=0. No sklearn dependency."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 2:** `from __future__ import annotations`

Tip annotasiyalarının işlənməsini təxirə salır; metrik hesabına təsir etmir.

**Sətir 3:** `import numpy as np`

Array və riyazi əməliyyatlar üçün NumPy-ni np qısa adı ilə gətirir.

**Sətir 6:** `def _paired_arrays(y_true, values):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 7:** `"""Validate inputs to prevent silent broadcasting or invalid results."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 8:** `y_true = np.asarray(y_true, dtype=float)`

Həqiqi cavabları NumPy array-inə və real ədədlərə çevirir. Array üzərində bütün elementlər üçün birlikdə hesablamaq mümkündür.

**Sətir 9:** `values = np.asarray(values, dtype=float)`

Proqnoz və ya score-ları da eyni ədədi formata çevirir.

**Sətir 10:** `if y_true.ndim != 1 or values.ndim != 1:`

Girişlər birölçülü olmalıdır. (n,1) ilə (n,) fərqli formalardır; nəzarətsiz hesab yanlış n×n nəticə yarada bilər.

**Sətir 11:** `raise ValueError("Inputs must be one-dimensional arrays.")`

Yanlış girişi aydın mesajla rədd edir; səhv rəqəm hesablayıb davam etmir.

**Sətir 12:** `if y_true.size == 0 or y_true.shape != values.shape:`

Boş məlumatı və cavab/proqnoz saylarının fərqini rədd edir.

**Sətir 13:** `raise ValueError("Inputs must be non-empty and have the same shape.")`

Yanlış girişi aydın mesajla rədd edir; səhv rəqəm hesablayıb davam etmir.

**Sətir 14:** `if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(values)):`

NaN və sonsuzluq varsa hesablamanı dayandırır.

**Sətir 15:** `raise ValueError("Inputs must contain only finite numbers.")`

Yanlış girişi aydın mesajla rədd edir; səhv rəqəm hesablayıb davam etmir.

**Sətir 16:** `return y_true, values`

Yoxlanmış iki array-i çağıran funksiyaya qaytarır.

**Sətir 19:** `def _binary_labels(labels):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 20:** `if not np.all((labels == 0) | (labels == 1)):`

Hər label ya 0, ya 1 olmalıdır. | elementlər üzrə OR deməkdir.

**Sətir 21:** `raise ValueError("Labels must be 0 (standard) or 1 (premium).")`

Yanlış girişi aydın mesajla rədd edir; səhv rəqəm hesablayıb davam etmir.

**Sətir 24:** `def rmse(y_true, y_pred) -> float:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 25:** `y_true, y_pred = _paired_arrays(y_true, y_pred)`

Ortaq yoxlama funksiyasını çağırır və yoxlanmış array-ləri götürür.

**Sətir 26:** `errors = y_true - y_pred`

Hər ev üçün eᵢ=yᵢ−ŷᵢ hesablayır.

**Sətir 27:** `return float(np.sqrt(np.mean(errors ** 2)))`

**2 kvadrat, mean orta, sqrt kökdür: birbaşa RMSE düsturudur. float NumPy ədədini Python ədədinə çevirir.

**Sətir 30:** `def mae(y_true, y_pred) -> float:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 31:** `y_true, y_pred = _paired_arrays(y_true, y_pred)`

Ortaq yoxlama funksiyasını çağırır və yoxlanmış array-ləri götürür.

**Sətir 32:** `return float(np.mean(np.abs(y_true - y_pred)))`

abs mütləq xətaları alır, mean ortanı hesablayır: MAE.

**Sətir 35:** `def r2(y_true, y_pred) -> float:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 36:** `"""One sample: NaN. Constant target: 1 if perfect, otherwise 0."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 37:** `y_true, y_pred = _paired_arrays(y_true, y_pred)`

Ortaq yoxlama funksiyasını çağırır və yoxlanmış array-ləri götürür.

**Sətir 38:** `if y_true.size < 2:`

İkidən az müşahidədə R² üçün kifayət qədər məlumat yoxdur.

**Sətir 39:** `return float("nan")`

Təyin olunmayan nəticəni NaN ilə bildirir; bu sıfır bal demək deyil.

**Sətir 40:** `ss_res = np.sum((y_true - y_pred) ** 2)`

Modelin kvadrat xətalarının cəmi: SS_res.

**Sətir 41:** `ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)`

Həqiqi qiymətlərin ortadan kvadrat uzaqlıqlarının cəmi: SS_tot.

**Sətir 42:** `if ss_tot == 0:`

Bütün həqiqi qiymətlər eynidirsə sıfıra bölünməyə xüsusi yanaşır.

**Sətir 43:** `return 1.0 if ss_res == 0 else 0.0`

Sabit hədəfdə mükəmməl nəticəyə 1, digərinə 0 verən konvensiyadır.

**Sətir 44:** `return float(1 - ss_res / ss_tot)`

R² düsturunu hesablayır.

**Sətir 47:** `def confusion_matrix(y_true, y_pred):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 48:** `"""Rows=true, columns=predicted: [[TN, FP], [FN, TP]]."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 49:** `y_true, y_pred = _paired_arrays(y_true, y_pred)`

Ortaq yoxlama funksiyasını çağırır və yoxlanmış array-ləri götürür.

**Sətir 50:** `_binary_labels(y_true)`

Həqiqi label-ların 0/1 olmasını yoxlayır.

**Sətir 51:** `_binary_labels(y_pred)`

Proqnoz label-larının 0/1 olmasını yoxlayır.

**Sətir 52:** `tn = np.sum((y_true == 0) & (y_pred == 0))`

Mötərizələrdə həqiqi və proqnoz sinif şərtlərini & (AND) ilə birləşdirir. np.sum uyğun gələn müşahidələri sayır; dəyişən həmin TN/FP/FN/TP sayıdır.

**Sətir 53:** `fp = np.sum((y_true == 0) & (y_pred == 1))`

Mötərizələrdə həqiqi və proqnoz sinif şərtlərini & (AND) ilə birləşdirir. np.sum uyğun gələn müşahidələri sayır; dəyişən həmin TN/FP/FN/TP sayıdır.

**Sətir 54:** `fn = np.sum((y_true == 1) & (y_pred == 0))`

Mötərizələrdə həqiqi və proqnoz sinif şərtlərini & (AND) ilə birləşdirir. np.sum uyğun gələn müşahidələri sayır; dəyişən həmin TN/FP/FN/TP sayıdır.

**Sətir 55:** `tp = np.sum((y_true == 1) & (y_pred == 1))`

Mötərizələrdə həqiqi və proqnoz sinif şərtlərini & (AND) ilə birləşdirir. np.sum uyğun gələn müşahidələri sayır; dəyişən həmin TN/FP/FN/TP sayıdır.

**Sətir 56:** `return np.array([[tn, fp], [fn, tp]], dtype=int)`

Sayları razılaşdırılmış 2×2 tam ədəd matrisində qaytarır.

**Sətir 59:** `def precision_recall_f1(y_true, y_pred):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 60:** `"""Positive-class metrics; undefined divisions return 0.0."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 61:** `matrix = confusion_matrix(y_true, y_pred)`

Dörd nəticə növünün sayını əvvəlki funksiyadan alır.

**Sətir 62:** `tn, fp, fn, tp = matrix.ravel()`

Matrisin sətirlərini ardıcıl düzləşdirib dörd sayı dəyişənlərə ayırır.

**Sətir 63:** `precision = float(tp / (tp + fp)) if tp + fp else 0.0`

TP/(TP+FP); heç premium proqnozu yoxdursa 0 qaytarır.

**Sətir 64:** `recall = float(tp / (tp + fn)) if tp + fn else 0.0`

TP/(TP+FN); həqiqi premium yoxdursa 0 qaytarır.

**Sətir 65:** `denominator = precision + recall`

F1 məxrəcini ayrıca saxlayır.

**Sətir 66:** `f1 = float(2 * precision * recall / denominator) if denominator else 0.0`

Harmonik orta düsturudur; P+R sıfırdırsa 0 qaytarır.

**Sətir 67:** `return precision, recall, f1`

Üç metriyi bu sıra ilə tuple kimi qaytarır.

**Sətir 70:** `def roc_auc(y_true, scores) -> float:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 71:** `"""Average-rank AUC. Larger scores mean premium; one class gives NaN.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 73:** `Pass SVM margins or premium probabilities, rather than predicted labels.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 74:** `"""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 75:** `y_true, scores = _paired_arrays(y_true, scores)`

Label və score saylarını, formalarını, sonlu olmasını yoxlayır.

**Sətir 76:** `_binary_labels(y_true)`

Həqiqi label-ların 0/1 olmasını yoxlayır.

**Sətir 77:** `positive = y_true == 1`

Premium müşahidələri True/False maskası ilə seçir.

**Sətir 78:** `n_positive = int(np.sum(positive))`

True-ları sayır: n₊.

**Sətir 79:** `n_negative = y_true.size - n_positive`

Ümumi saydan pozitivləri çıxır: n₋.

**Sətir 80:** `if n_positive == 0 or n_negative == 0:`

İki sinifdən biri yoxdursa cüt müqayisəsi və AUC mümkün deyil.

**Sətir 81:** `return float("nan")`

Təyin olunmayan nəticəni NaN ilə bildirir; bu sıfır bal demək deyil.

**Sətir 82:** `order = np.argsort(scores, kind="stable")`

Score-ları kiçikdən böyüyə düzən indeksləri tapır; score-ları özü dəyişmir.

**Sətir 83:** `sorted_scores = scores[order]`

Həmin indekslərlə sıralanmış score-ları alır.

**Sətir 84:** `ranks = np.empty(scores.size, dtype=float)`

Hər müşahidənin rank-ını saxlayacaq array ayırır; aşağıda bütün elementlər doldurulur.

**Sətir 85:** `start = 0`

İlk bərabər-score qrupunun başlanğıc indeksidir.

**Sətir 86:** `while start < scores.size:`

Bütün score qruplarını bir-bir emal edir.

**Sətir 87:** `end = start + 1`

Qrupun bitdiyi, daxil olmayan indeks əvvəlcə növbəti mövqedir.

**Sətir 88:** `while end < scores.size and sorted_scores[end] == sorted_scores[start]:`

Növbəti score eynidirsə qrup davam edir.

**Sətir 89:** `end += 1`

Bərabər score qrupunu bir mövqe genişləndirir.

**Sətir 90:** `average_rank = ((start + 1) + end) / 2`

İndekslər 0-dan, rank-lar 1-dən başlayır. Qrupun ilk və son rank-ının ortasını tapır.

**Sətir 91:** `ranks[order[start:end]] = average_rank`

Orta rank-ı qrupdakı müşahidələrin ORİJİNAL mövqelərinə yazır; label uyğunluğu qorunur.

**Sətir 92:** `start = end`

Növbəti score qrupuna keçir.

**Sətir 93:** `rank_sum = np.sum(ranks[positive])`

Yalnız premium müşahidələrin rank-larını toplayır: R₊.

**Sətir 94:** `wins = rank_sum - n_positive * (n_positive + 1) / 2`

Pozitivlərin öz aralarında rank payını çıxır; neqativlərə qarşı qələbə sayını saxlayır, bərabərliklər yarımdır.

**Sətir 95:** `return float(wins / (n_positive * n_negative))`

Qələbələri bütün premium-standard cütlərinin sayına bölür: AUC.

**Sətir 98:** `def compare(name_to_metrics: dict) -> None:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 99:** `"""Print {model_name: {scalar_metric: value}}; missing values are n/a.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 101:** `Print confusion matrices separately. Use identical splits for comparisons.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 102:** `"""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 103:** `if not name_to_metrics:`

Nəticə lüğəti boşdursa ayrıca mesaj verir.

**Sətir 104:** `print("No model results available.")`

Mötərizədəki mesajı və ya hesablanmış nəticəni ekranda göstərir; modeli öyrətmir.

**Sətir 105:** `return`

Funksiyanı burada bitirir; aşağıdakı sətirlər işləməz.

**Sətir 106:** `metric_names = list(dict.fromkeys(key for metrics in name_to_metrics.values() for key in metrics))`

Bütün model lüğətlərindən metrik adlarını çıxarır; təkrarları silir, ilk görünmə sırasını qoruyur.

**Sətir 107:** `rows = []`

Cədvəl sətirləri üçün boş siyahı yaradır.

**Sətir 108:** `for name, metrics in name_to_metrics.items():`

Hər model adını və onun metrik lüğətini növbə ilə götürür.

**Sətir 109:** `row = [str(name)]`

Sətirin ilk xanasına model adını yazır.

**Sətir 110:** `for metric in metric_names:`

Hər metrik sütununu doldurur.

**Sətir 111:** `value = metrics.get(metric)`

Metrik varsa qiymətini, yoxdursa None alır.

**Sətir 112:** `row.append("n/a" if value is None else f"{float(value):.4f}")`

Olmayan metriyə n/a, olan metriyə dörd onluq rəqəmlə yazılış verir. Hesabın özü yuvarlaqlaşdırılmır.

**Sətir 113:** `rows.append(row)`

Hazır sətiri cədvəl siyahısına əlavə edir.

**Sətir 114:** `headers = ["Model", *metric_names]`

Başlıq sətirini qurur; * siyahının elementlərini açır.

**Sətir 115:** `widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]`

Hər sütun üçün ən uzun xananın mətn uzunluğunu tapır.

**Sətir 116:** `def format_row(row):`

Bir sətiri ekranda düz sütunlarla göstərəcək köməkçi funksiya yaradır.

**Sətir 117:** `return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))`

Xanaları sağdan boşluqla doldurur və | ilə birləşdirir.

**Sətir 118:** `print(format_row(headers))`

Başlıqları ekrana çıxarır.

**Sətir 119:** `print("-+-".join("-" * width for width in widths))`

Başlığın altına ayırıcı xətt çəkir.

**Sətir 120:** `for row in rows:`

Hazır nəticə sətirlərini gəzər.

**Sətir 121:** `print(format_row(row))`

Modelin nəticə sətirini ekrana çıxarır.


### src/run_all.py

**Sətir 1:** `"""Initial skeleton. Run examples with: python -m src.run_all --demo.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 2:** `Synthetic examples are not final project results.`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 3:** `"""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 4:** `from __future__ import annotations`

Tip annotasiyalarının işlənməsini təxirə salır; metrik hesabına təsir etmir.

**Sətir 5:** `import argparse`

Komanda sətri seçimlərini oxuyan Python modulunu gətirir.

**Sətir 6:** `from . import evaluate`

Eyni src paketindəki evaluation modulunu gətirir.

**Sətir 9:** `def regression_metrics(y_true, y_pred):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 10:** `"""Both arrays must be on the same scale: price or log-price."""`

Docstring-in izah sətridir; giriş qaydasını və ya xüsusi hal davranışını sənədləşdirir, riyazi əməliyyat etmir.

**Sətir 11:** `return {`

Ad→qiymət lüğətini qaytarmağa başlayır.

**Sətir 12:** `"RMSE": evaluate.rmse(y_true, y_pred),`

Soldakı metrik adının qarşısına sağdakı funksiyanın hesabladığı qiyməti qoyur.

**Sətir 13:** `"MAE": evaluate.mae(y_true, y_pred),`

Soldakı metrik adının qarşısına sağdakı funksiyanın hesabladığı qiyməti qoyur.

**Sətir 14:** `"R2": evaluate.r2(y_true, y_pred),`

Soldakı metrik adının qarşısına sağdakı funksiyanın hesabladığı qiyməti qoyur.

**Sətir 15:** `}`

Lüğətin bitdiyini bildirir.

**Sətir 18:** `def classification_metrics(y_true, y_pred, scores):`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 19:** `precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)`

Üç metriyi hesablayıb ayrı dəyişənlərə yerləşdirir.

**Sətir 20:** `return {`

Ad→qiymət lüğətini qaytarmağa başlayır.

**Sətir 21:** `"Precision": precision,`

Hesablanmış metriyi uyğun adla lüğətə yerləşdirir.

**Sətir 22:** `"Recall": recall,`

Hesablanmış metriyi uyğun adla lüğətə yerləşdirir.

**Sətir 23:** `"F1": f1,`

Hesablanmış metriyi uyğun adla lüğətə yerləşdirir.

**Sətir 24:** `"ROC-AUC": evaluate.roc_auc(y_true, scores),`

Soldakı metrik adının qarşısına sağdakı funksiyanın hesabladığı qiyməti qoyur.

**Sətir 25:** `}`

Lüğətin bitdiyini bildirir.

**Sətir 28:** `def demo():`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 29:** `print("SYNTHETIC EXAMPLES ONLY: these are not trained-model results.")`

Mötərizədəki mesajı və ya hesablanmış nəticəni ekranda göstərir; modeli öyrətmir.

**Sətir 30:** `evaluate.compare({"Regression example": regression_metrics([10, 20, 30], [12, 18, 33])})`

Metrik köməkçisindən nəticə lüğəti alır və müqayisə cədvəlinə verir.

**Sətir 31:** `y_true = [0, 0, 1, 1]`

Demo üçün iki standard, iki premium həqiqi label yaradır.

**Sətir 32:** `y_pred = [0, 1, 0, 1]`

Demo proqnozları: bir FP və bir FN var.

**Sətir 33:** `scores = [0.1, 0.4, 0.35, 0.8]`

Label-lardan ayrıca demo score-larıdır; threshold .5 ilə eyni demo label-ları yaratması tələb edilmir.

**Sətir 34:** `evaluate.compare({"Classification example": classification_metrics(y_true, y_pred, scores)})`

Metrik köməkçisindən nəticə lüğəti alır və müqayisə cədvəlinə verir.

**Sətir 35:** `print("Confusion matrix [[TN, FP], [FN, TP]]:")`

Mötərizədəki mesajı və ya hesablanmış nəticəni ekranda göstərir; modeli öyrətmir.

**Sətir 36:** `print(evaluate.confusion_matrix(y_true, y_pred))`

Mötərizədəki mesajı və ya hesablanmış nəticəni ekranda göstərir; modeli öyrətmir.

**Sətir 39:** `def main() -> None:`

Bu adda funksiya yaradır; mötərizədə qəbul etdiyi girişlərdir. -> float/None varsa qaytarılan nəticənin tipini sənədləşdirir.

**Sətir 40:** `parser = argparse.ArgumentParser(description=__doc__)`

CLI parser yaradır və modulun izahını help mətninə verir.

**Sətir 41:** `parser.add_argument("--demo", action="store_true", help="Run synthetic metric examples.")`

--demo verildikdə True olacaq seçim əlavə edir.

**Sətir 42:** `args = parser.parse_args()`

İstifadəçinin komanda sətri seçimlərini oxuyur.

**Sətir 43:** `if args.demo:`

Demo seçilibsə nümunə yoluna keçir.

**Sətir 44:** `demo()`

Sintetik nümunələri işlədir.

**Sətir 45:** `return`

Funksiyanı burada bitirir; aşağıdakı sətirlər işləməz.

**Sətir 46:** `# Team integration order:`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 47:** `# 1. Load/clean; split once; fit preprocessing on training only.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 48:** `# 2. Derive tier threshold from training prices; apply to val/test.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 49:** `# 3. Tune using validation; train matched sklearn baselines.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 50:** `# 4. Obtain test predictions and scores in unchanged sample order.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 51:** `# 5. Compute metrics, CV uncertainty, timing and error analysis.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 52:** `# 6. Save labelled report figures and print comparison tables.`

İnteqrasiya addımını izah edən şərhdir; Python bunu icra etmir.

**Sətir 53:** `parser.exit(status=2, message="Full pipeline pending team data/model integration. Use --demo for examples.")`

Tam pipeline hələ hazır olmadığı üçün status 2 ilə aydın mesaj verir. Bu ilkin skeletdir; yekun təqdimatda real pipeline ilə əvəz olunmalıdır.

**Sətir 56:** `if __name__ == "__main__":`

Fayl icra olunanda main çağırılır; import ediləndə avtomatik demo/pipeline işləmir.

**Sətir 57:** `main()`

Komanda sətrinin giriş funksiyasını çağırır.

## Yoxlama və davamı

Daxili layihə qovluğundan: `python -m unittest discover -s tests -v` və `python -m src.run_all --demo`.

Testlər əvvəlcədən hesablanmış nəticələri, AUC üçün alternativ bütün cütləri müqayisə üsulunu, bərabər score-ları, etibarsız girişləri və xüsusi halları yoxlayır. sklearn varsa ayrıca müqayisə testi də işləyir. Hazırkı bundled mühitdə sklearn olmadığı üçün həmin müqayisə testi skip edilib; digər 4 test keçib.

Real nəticələr üçün data və model modulları lazımdır. Yekunda validation əsasında konfiqurasiya seçimi, eyni test bölgüsündə sklearn müqayisəsi, CV mean±std, fit/predict vaxtları, confusion matrix/əyrilər, feature importance/SVM çəkiləri və real səhv analizi birlikdə tamamlanmalıdır. Bu işlərin məsuliyyət bölgüsü komanda ilə razılaşdırılmalıdır. Test məlumatına baxıb hiperparametr seçmək olmaz.

Bu kod və izah hazırlanarkən AI köməyindən istifadə olunub; layihə reportunun Tools and Acknowledgements hissəsində dürüst açıqlanmalıdır.
