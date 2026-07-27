import numpy as np
from scipy.stats import shapiro, kruskal, mannwhitneyu


def check_normality(values: list[float], name: str) -> dict:
    stat, p = shapiro(values)
    normal = p > 0.05
    return {
        "variable": name,
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "shapiro_w": round(stat, 4),
        "shapiro_p": round(p, 4),
        "is_normal": normal,
        "interpretation": "Normal" if normal else "No normal",
    }


def compare_algorithms(data: dict[str, list[float]]) -> dict:
    names = list(data.keys())
    groups = [data[n] for n in names]

    normality_results = {n: check_normality(data[n], n) for n in names}
    all_normal = all(r["is_normal"] for r in normality_results.values())

    if len(groups) >= 2:
        if all_normal:
            from pingouin import anova
            df_list = []
            for n in names:
                for v in data[n]:
                    df_list.append({"algorithm": n, "value": v})
            import pandas as pd
            df = pd.DataFrame(df_list)
            aov = anova(dv="value", between="algorithm", data=df, detailed=True)
            test_name = "ANOVA"
            test_stat = float(aov["F"].values[0])
            test_p = float(aov["p_unc"].values[0])
            effect = float(aov["np2"].values[0])
            effect_name = "np2"
        else:
            h_stat, p_val = kruskal(*groups)
            test_name = "Kruskal-Wallis"
            test_stat = round(h_stat, 4)
            test_p = round(p_val, 4)
            effect = None
            effect_name = None

        posthoc = []
        if test_p < 0.05:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    u_stat, p_pair = mannwhitneyu(data[names[i]], data[names[j]])
                    from pingouin import compute_effsize
                    d = compute_effsize(data[names[i]], data[names[j]], eftype="cohen")
                    posthoc.append({
                        "comparison": f"{names[i]} vs {names[j]}",
                        "mannwhitney_u": int(u_stat),
                        "p_value": round(p_pair, 4),
                        "cohens_d": round(float(d), 3),
                    })
    else:
        test_name = "N/A"
        test_stat = None
        test_p = None
        effect = None
        effect_name = None
        posthoc = []

    return {
        "normality": normality_results,
        "all_normal": all_normal,
        "test_name": test_name,
        "test_statistic": round(test_stat, 4) if test_stat is not None else None,
        "p_value": round(test_p, 4) if test_p is not None else None,
        "significant": test_p is not None and test_p < 0.05,
        "effect_size": round(effect, 4) if effect is not None else None,
        "effect_name": effect_name,
        "posthoc": posthoc,
    }


def format_report(result: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("ANALISIS ESTADISTICO")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Normalidad (Shapiro-Wilk):")
    for n, r in result["normality"].items():
        lines.append(f"  {n}: W={r['shapiro_w']:.3f}, p={r['shapiro_p']:.3f} -> {r['interpretation']}")
    lines.append(f"  Todos normales: {'Si' if result['all_normal'] else 'No'}")
    lines.append("")

    lines.append(f"Prueba: {result['test_name']}")
    if result["test_statistic"] is not None:
        lines.append(f"  Estadistico: {result['test_statistic']}")
        lines.append(f"  p-valor: {result['p_value']}")
        lines.append(f"  Significativo (p<0.05): {'Si' if result['significant'] else 'No'}")
    if result["effect_size"] is not None:
        lines.append(f"  Tamano del efecto ({result['effect_name']}): {result['effect_size']}")
    lines.append("")

    if result["posthoc"]:
        lines.append("Comparaciones post-hoc (Mann-Whitney + Cohen's d):")
        for p in result["posthoc"]:
            lines.append(f"  {p['comparison']}: U={p['mannwhitney_u']}, p={p['p_value']}, d={p['cohens_d']}")
    lines.append("=" * 60)
    return "\n".join(lines)
