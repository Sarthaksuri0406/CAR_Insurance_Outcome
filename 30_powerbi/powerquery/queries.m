let
    Source = Csv.Document(File.Contents("../data/06_bi_exports/facts_claims_scored.csv"),[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    Promoted
