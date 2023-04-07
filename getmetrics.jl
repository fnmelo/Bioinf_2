#!/usr/local/bin/julia
#title           :getmetrics.jl
#description     :Calculate performance metrics for given files
#author          :Carlos Vigil Vásquez
#date            :20230202
#version         :20230202a
#copyright       :Copyright (C) 2022 Carlos Vigil Vásquez (carlos.vigil.v@gmail.com)
#license         :Permission to copy and modify is granted under the MIT license

using ArgParse
using Base.Threads
using CSV
using DataFrames
using Glob
using MLBase
using ProgressMeter
using SimSpread

function getperformance(
    df::DataFrame;
    ycol::String="TP",
    yhatcol::String="Score"
)
    # Initialize function
    score = convert(Vector{Float64}, df[!, yhatcol])
    tp = convert(Vector{Bool}, df[!, ycol])

    # Generate confusion matrix and common metrics
    thresholds = sort(unique(score))

    confusion = roc(tp, score, thresholds)
    recalls = SimSpread.recall.(confusion)
    precisions = SimSpread.precision.(confusion)
    tprs = true_negative_rate.(confusion)
    fprs = false_negative_rate.(confusion)

    # Standard metrics
    auprc = SimSpread.AuPRC(tp, score)
    auroc = SimSpread.AuROC(tp, score)

    # Early recognition
    bedroc = SimSpread.BEDROC(tp, score)
    r20 = SimSpread.performanceatL(tp, score, SimSpread.recall)
    p20 = SimSpread.performanceatL(tp, score, SimSpread.precision)

    # Binary metrics
    f1 = SimSpread.maxperformance(confusion, SimSpread.f1score)
    bacc = SimSpread.maxperformance(confusion, SimSpread.balancedaccuracy)
    acc = SimSpread.maxperformance(confusion, SimSpread.accuracy)
    #mcc = SimSpread.maxperformance(confusion, SimSpread.mcc)

    return DataFrame(
        groupid=[missing],
        AuROC=[auroc],
        AuPRC=[auprc],
        BEDROC=[bedroc],
        r20 = [r20],
        p20 = [p20],
        f1=[f1],
        bacc=[bacc],
        acc=[acc],
        #mcc = [mcc],
    )
end

function getperformance(
    df::DataFrame,
    groupby::String;
    ycol::String="TP",
    yhatcol::String="Score"
)
    # Initialize function
    grouper = df[!, groupby]
    alltp = convert(Vector{Bool}, df[!, ycol])
    allscore = convert(Vector{Float64}, df[!, yhatcol])

    # Evaluate performance per group
    groupperformance = DataFrame()
    for groupᵢ in unique(grouper)
        # Get group scores and true-positive state
        tp = alltp[grouper.==groupᵢ]
        score = allscore[grouper.==groupᵢ]

        # Confusion matrix
        thresholds = sort(unique(score))

        confusion = roc(tp, score, thresholds)
        recalls = SimSpread.recall.(confusion)
        precisions = SimSpread.precision.(confusion)
        tprs = true_negative_rate.(confusion)
        fprs = false_negative_rate.(confusion)

        # Standard metrics
        auprc = SimSpread.AuPRC(tp, score)
        auroc = SimSpread.AuROC(tp, score)

        # Early recognition
        bedroc = SimSpread.BEDROC(tp, score)
        # r20 = SimSpread.performanceatL(tp, score, SimSpread.recall)
        # p20 = SimSpread.performanceatL(tp, score, SimSpread.precision)

        # Binary metrics
        f1 = SimSpread.maxperformance(confusion, SimSpread.f1score)
        bacc = SimSpread.maxperformance(confusion, SimSpread.balancedaccuracy)
        acc = SimSpread.maxperformance(confusion, SimSpread.accuracy)
        #mcc = SimSpread.maxperformance(confusion, SimSpread.mcc)

        append!(
            groupperformance,
            DataFrame(
                groupid=[groupᵢ],
                AuROC=[auroc],
                AuPRC=[auprc],
                BEDROC=[bedroc],
                # r20 = [r20],
                # p20 = [p20],
                f1=[f1],
                bacc=[bacc],
                # acc=[acc],
                #mcc = [mcc],
            )
        )
    end

    return groupperformance
end


function main(args)
    # Argument parsing
    settings = ArgParseSettings()

    GROUP_CHOICES = ["pooled", "fold", "ligands", "targets"]
    GROUP_MAPPER = Dict("pooled" => missing, "fold" => "Fold", "ligands" => "LigID", "targets" => "TrgID")

    add_arg_group!(settings, "I/O option:")
    @add_arg_table! settings begin
        "-i"
        help = "input"
        arg_type = String
        action = :store_arg
        required = true
        "--groupby"
        help = "Group prediction; must be one of " * join(GROUP_CHOICES, ", ", " or ")
        arg_type = String
        action = :store_arg
        default = "pooled"
        range_tester = (x -> x ∈ GROUP_CHOICES)
        required = false
    end

    args = parse_args(args, settings)
    inputfiles = args["i"]
    @show grouping = GROUP_MAPPER[args["groupby"]]

    # Main body
    inputfiles = Glob.glob(inputfiles)
    pbar = Progress(length(inputfiles))
    @show length(inputfiles)
    @threads for outfile in inputfiles
        # Extract metadata from filename
        scenario, parameters, _ = split(basename(outfile), '.')
        dataset, cv = split(scenario, '_')
        fingerprint, measure, cutoff, iteration = split(parameters, '_')
        cutoff = parse(Float64, string(cutoff[1] * "." * cutoff[2:end]))

        # Load data to DataFrame
        out = CSV.read(
            outfile,
            DataFrame;
            header=["Fold", "LigID", "TrgID", "Score", "TP"],
            delim=","
        )
        filter!(:Score => s -> s != -99, out)

        # Append performances to list
        if typeof(grouping) == String
            performance = getperformance(out, grouping)
        else
            performance = getperformance(out)
        end
        performance[!, :dataset] .= dataset
        performance[!, :cv] .= cv
        performance[!, :iteration] .= iteration
        performance[!, :fingerprint] .= fingerprint
        performance[!, :measure] .= measure
        performance[!, :cutoff] .= cutoff

        CSV.write(outfile * "." * args["groupby"] *"_performance.csv", performance, header=true)
        next!(pbar)
    end
end

main(ARGS)

