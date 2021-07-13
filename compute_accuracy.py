import geopandas as gpd
import pandas as pd
import numpy as np


def read_points(computed_fname, surveyed_fname):
    computed_points_gdf = gpd.read_file(computed_fname)
    computed_points = pd.DataFrame(computed_points_gdf.drop(columns="geometry"))
    computed_points = computed_points.assign(X=computed_points_gdf.geometry.x, Y=computed_points_gdf.geometry.y)

    surveyed_points = pd.read_csv(surveyed_fname, names=["GCP", "X", "Y", "Z"])

    return computed_points, surveyed_points


def compute_statistics(va):
    nv = len(va)
    dims = len(va[0])
    stats = np.empty(shape=(5, dims))
    stats[0, :] = np.min(va, axis=0)
    stats[1, :] = np.max(va, axis=0)
    stats[2, :] = np.average(va, axis=0)
    stats[3, :] = np.std(va, axis=0)
    stats[4, :] = np.sqrt(np.sum(np.square(va), axis=0) / (nv - 1))

    row_indices = ["Minimum", "Maximum", "Average", "Std.", "RMSE"]
    if dims == 2:
        column_names = ["X_diff", "Y_diff"]
    else:
        column_names = ["X_diff", "Y_diff", "Z_diff"]

    stats_df = pd.DataFrame(stats, index=row_indices, columns=column_names)
    total_rmse = np.sqrt(np.sum(np.square(stats[4, :]), axis=0))

    return stats_df, total_rmse


def absolute_accuracy(computed, survyed, by_gcp=True):
    print("\n*********************\n  Absolute accuracy   \n*********************")
    ## Compute differences by each GCP w.r.t. surveyed data
    X_diff_abs = []
    Y_diff_abs = []
    RMSE_abs = []
    for row in computed.values:
        for row2 in survyed.values:
            if row[0] == row2[0]:
                X_diff_abs.append(row[2] - row2[1])
                Y_diff_abs.append(row[3] - row2[2])
                RMSE_abs.append(np.sqrt(np.sum(np.square(np.array([X_diff_abs[-1], Y_diff_abs[-1]])), axis=0)))

    computed_points = computed.assign(X_diff_abs=X_diff_abs, Y_diff_abs=Y_diff_abs, RMSE_abs=RMSE_abs)
    # print(computed_points)

    ## Compute whole stats
    nv_whole = len(computed_points)
    X_diff_abs_whole = np.array(computed_points.X_diff_abs, dtype=np.float16)
    Y_diff_abs_whole = np.array(computed_points.Y_diff_abs, dtype=np.float16)
    va_abs_whole = np.array([X_diff_abs_whole, Y_diff_abs_whole]).T
    stats_abs_whole_df, total_rmse_abs_whole = compute_statistics(va_abs_whole)

    ## Compute stats by GCP
    if by_gcp:
        print(f" *********************\n * Individual markers\n *********************")
        for k, v in computed_points.groupby(["GCP"]).indices.items():
            rows = computed_points.loc[v]
            nv = len(rows)

            gcp = np.average(rows.GCP.values)
            X_diff_abs_each = rows.X_diff_abs.values
            Y_diff_abs_each = rows.Y_diff_abs.values
            va_abs_each = np.array([X_diff_abs_each, Y_diff_abs_each]).T
            stats_abs_each_df, total_rmse_abs_each = compute_statistics(va_abs_each)

            print(f" * Marker {int(gcp)}: {nv}")
            print(stats_abs_each_df)
            print(f" * Total RMSE: {total_rmse_abs_each:.2f} m\n")

    print(f" *********************\n * Whole markers: {nv_whole}\n *********************")
    print(stats_abs_whole_df)
    print(f" * Total RMSE: {total_rmse_abs_whole:.2f} m")
    stats_abs_whole_df.to_csv("coords_accuracy/absolute_whole.csv")


def relative_accuracy(computed, by_gcp=True):
    print("\n*********************\n  Relative accuracy   \n*********************")
    ## Compute stats by GCP

    print(f" *********************\n * Individual markers\n *********************")
    coords_avg_index = []
    for k, v in computed.groupby(["GCP"]).indices.items():
        rows = computed.loc[v]
        nv = len(rows)

        gcp = np.average(rows.GCP.values)
        X_avg_each = np.average(rows.X.values)
        Y_avg_each = np.average(rows.Y.values)

        coords_avg_index.append([int(gcp), X_avg_each, Y_avg_each])

        if by_gcp:
            X_diff_rel_each = rows.X.values - X_avg_each
            Y_diff_rel_each = rows.Y.values - Y_avg_each
            va_rel_each = np.array([X_diff_rel_each, Y_diff_rel_each]).T
            stats_rel_each_df, total_rmse_rel_each = compute_statistics(va_rel_each)

            print(f" * Marker {int(gcp)}: {nv}")
            print(stats_rel_each_df)
            print(f" * Total RMSE: {total_rmse_rel_each:.2f} m\n")

    X_diff_rel = []
    Y_diff_rel = []
    RMSE_rel = []
    for row in computed.values:
        for row2 in coords_avg_index:
            if row[0] == row2[0]:
                X_diff_rel.append(row[2] - row2[1])
                Y_diff_rel.append(row[3] - row2[2])
                RMSE_rel.append(np.sqrt(np.sum(np.square(np.array([X_diff_rel[-1], Y_diff_rel[-1]])), axis=0)))

    computed_points = computed.assign(X_diff_rel=X_diff_rel, Y_diff_rel=Y_diff_rel, RMSE_rel=RMSE_rel)
    # print(computed_points)

    ## Compute whole stats
    nv = len(computed_points)
    X_diff_rel_whole = np.array(computed_points.X_diff_rel, dtype=np.float16)
    Y_diff_rel_whole = np.array(computed_points.Y_diff_rel, dtype=np.float16)
    va_rel_whole = np.array([X_diff_rel_whole, Y_diff_rel_whole]).T
    stats_rel_whole_df, total_rmse_rel_whole = compute_statistics(va_rel_whole)

    print(f" *********************\n * Whole markers: {nv}\n *********************")
    print(stats_rel_whole_df)
    print(f" * Total RMSE: {total_rmse_rel_whole:.2f} m")
    stats_rel_whole_df.to_csv("coords_accuracy/relative_whole.csv")


if __name__ == "__main__":
    # pd.set_option('display.max_columns', None)
    pd.options.display.float_format = "{:,.2f}".format

    computed_fname = "coords_accuracy/computed_gp.geojson"  # geojson
    surveyed_fname = "모슬포_GCP.csv"                        # csv

    computed_points, surveyed_points = read_points(computed_fname, surveyed_fname)
    absolute_accuracy(computed_points, surveyed_points, by_gcp=False)
    relative_accuracy(computed_points, by_gcp=False)

