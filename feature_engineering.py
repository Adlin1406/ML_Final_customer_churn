def create_features(df):

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["IsMonthly"] = (df["Contract"] == "Month-to-month").astype(int)

    df["HighValueCustomer"] = (df["MonthlyCharges"] > 80).astype(int)

    return df