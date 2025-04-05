# ----------------------------------------------------------------------
# Anish Rao
# 20066423
# Unsupervised LSTM Anomaly Detection on NAB Dataset
# Forecasting + Conformal Prediction Classifier
# ----------------------------------------------------------------------

# import Pkg
# packages = ["BSON", "CSV", "DataFrames", "DelimitedFiles", 
# "Flux", "JSON", "Statistics"]
# for pkg in packages
#     Pkg.add(pkg)
# end
# Pkg.update()

using BSON: @save
using CSV
using DataFrames
using Dates
using DelimitedFiles
using Flux
using JSON
using Random
using Statistics

Random.seed!(86)
```
---------------------------------- PARAMETER LOADING ----------------------------------
```
function load_params(json_path::String)::Dict{String, Dict{String, Float64}}
    if isfile(json_path)
        return JSON.parsefile(json_path)
    else
        error("params_unsupervised.json not found at $json_path")
    end
end
```
---------------------------------- DATA LOADING ----------------------------------
```
function load_csv_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    timestamps = DateTime.(df.timestamp, dateformat"yyyy-MM-dd HH:MM:SS")
    values = Float32.(df.value)
    return timestamps, values
end
```
---------------------------------- DATA NORMALIZATION and SPLITTING ----------------------------------
```
function split_and_normalize(values::Vector{Float32}, prob_ratio::Float64, calib_ratio::Float64)
    n = length(values)
    prob_count = Int(floor(prob_ratio * n))
    calib_count = Int(floor(calib_ratio * prob_count))
    train_count = prob_count - calib_count

    probation_vals = values[1:prob_count] 
    mean_prob = mean(probation_vals)
    std_prob = std(probation_vals)
    normal_values = (values .- mean_prob) ./ std_prob

    train_data = normal_values[1:train_count]
    calib_data = normal_values[train_count+1 : prob_count]
    test_data  = normal_values[prob_count+1 : end]
    
    return (;train_data, calib_data, test_data)
end   
```
---------------------------------- DATA SEQUENCING ----------------------------------
```
function create_sequences(values::Vector{Float32}, window_size::Int)
    n_seq = length(values) - window_size
    x = Array{Float32}(undef, window_size, 1, n_seq)
    y = Array{Float32}(undef, 1, n_seq)

    for i in 1:n_seq
        x[:, 1, i] = values[i : i + window_size - 1]
        y[1, i] = values[i + window_size]
    end

    return x, y
end

```
---------------------------------- MODEL TRAINING ----------------------------------
```
function build_lstm_model(window_size::Int, dropout::Float64)
    return Chain(
        LSTM(window_size => 64),
        Dropout(dropout),
        Dense(64 => 1)
    )
end

function train_model(x_train::Array{Float32, 3}, y_train::Array{Float32,2};
    window_size::Int, epochs::Int, lr::Float64, dropout::Float64)

    model = build_lstm_model(window_size, dropout)
    opt_state = Flux.setup(ADAMW(lr), model)

    for epoch in 1:epochs
        Flux.reset!(model)

        loss, grads = Flux.withgradient(model) do m
            y_pred = m(x_train)
            y_pred = dropdims(y_pred, dims=2)
            Flux.Losses.mse(y_pred, y_train)
        end

        Flux.Optimise.update!(opt_state, model, grads[1])
        println("Epoch $epoch -- Loss = $(round(loss, digits=4))")
    end

    return model
end
```
---------------------------------- CONFORMAL CLASSIFICATION  ----------------------------------
```
function conformal_classification(model, x_calib, y_calib; confidence::Float64)
    residuals = Float32[]

    for i in 1:size(x_calib, 3)
        x = x_calib[:, :, i]
        Flux.reset!(model)
        y_pred = model(x)
        push!(residuals, abs(y_pred[1] - y_calib[1, i]))
    end

    if any(isnan, residuals)
        return NaN, residuals
    end

    threshold = quantile(residuals, confidence)

    println("Threshold = ", round(threshold, digits=4))
    return threshold, residuals
end
```
---------------------------------- EVALUATION ----------------------------------
```
function evaluate_model(model, x_eval, y_eval, threshold)
    residuals = Float32[]
    predictions = Float32[]
    anomaly_flags = Int[]

    for i in 1:size(x_eval, 3)
        x = x_eval[:, :, i]
        Flux.reset!(model)
        y_pred = model(x)[1]
        res = abs(y_pred - y_eval[1, i])

        push!(residuals, res)
        push!(predictions, y_pred)
        push!(anomaly_flags, res > threshold ? 1 : 0)
    end

    return (;predictions, residuals, anomaly_flags)
end

function load_true_labels(label_path::String, file_key::String,
    eval_timestamps::Vector{DateTime})

    label_data = JSON.parsefile(label_path)
    windows = label_data[file_key]

    datetime_format = dateformat"yyyy-MM-dd HH:MM:SS.ssss"
    anomaly_windows = [(DateTime(w[1], datetime_format), 
    DateTime(w[2], datetime_format)) for w in windows]

    true_flags = Int[]
    for ts in eval_timestamps
        is_anomaly = any(r -> ts ≥ r[1] && ts ≤ r[2], anomaly_windows)
        push!(true_flags, is_anomaly ? 1 : 0)
    end

    return true_flags
end

function compute_metrics(filename::String, pred_flags::Vector{Int}, true_flags::Vector{Int})
    TP = sum(z -> z[1] == 1 && z[2] == 1, zip(pred_flags, true_flags))
    FP = sum(z -> z[1] == 1 && z[2] == 0, zip(pred_flags, true_flags))
    FN = sum(z -> z[1] == 0 && z[2] == 1, zip(pred_flags, true_flags))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-8)

    println("\nEvaluation Metrics for $filename")
    println("TP = $TP | FP = $FP | FN = $FN")
    println("Precision = $(round(precision * 100, digits=2))%")
    println("Recall    = $(round(recall * 100, digits=2))%")
    println("F1 Score  = $(round(f1 * 100, digits=2))%")

    return (;precision, recall, f1, TP, FP, FN)
end

```
---------------------------------- SAVING MODEL and METRICS ----------------------------------
```
function save_model(model, filename::String)
    mkpath("models")
    model_path = joinpath("models", "model_" * replace(filename, "/" => "_") * ".bson")
    @save model_path model
end

function save_metrics(filename::String, metrics::NamedTuple)
    mkpath("results")
    results_file = joinpath("results", "unsupervsied_metrics.csv")

    header = ["filename", "TP", "FP", "FN", "Precision (%)", "Recall (%)", "F1 Score (%)"]

    row = [
        filename, metrics.TP, metrics.FP, metrics.FN,
        round(metrics.precision * 100, digits=2),
        round(metrics.recall * 100, digits=2),
        round(metrics.f1 * 100, digits=2)
    ]

    if isfile(results_file)
        open(results_file, "a") do io
            writedlm(io, [row], ',')
        end
    else
        open(results_file, "w") do io
            writedlm(io, [header], ',')
            writedlm(io, [row], ',')
        end
    end
end
```
---------------------------------- IMPLEMENTATION ----------------------------------
```
data_root = "data"
label_path = "labels/combined_windows.json"
param_path = "params_unsupervised.json"

params = load_params(param_path)

all_files = sort(collect(keys(params)))

for filename in all_files
    println("\nProcessing: $filename")

    file_path = joinpath(data_root, filename)

    # Load data and parameters
    ts, values = load_csv_data(file_path)
    config = params[filename]
    probation_ratio = config["probation_ratio"]
    calib_ratio = config["calib_ratio"]
    window_size = Int(config["window_size"])
    epochs = Int(config["epochs"])
    lr = Float64(config["lr"])
    dropout = Float64(config["dropout"])
    conf = Float64(config["conf"])

    # Split and normalize data
    train_data, calib_data, eval_data = split_and_normalize(values, probation_ratio, calib_ratio)

    # Create sequences
    x_train, y_train = create_sequences(train_data, window_size)
    x_calib, y_calib = create_sequences(calib_data, window_size)
    x_eval,  y_eval  = create_sequences(eval_data,  window_size)

    # Train model
    model = train_model(
        x_train, y_train;
        window_size=window_size,
        epochs=epochs,
        lr=lr,
        dropout=dropout
    )

    # Classification
    threshold, calib_residuals = conformal_classification(
        model, x_calib, y_calib; confidence=conf)

    if !isfinite(threshold)
        save_metrics(filename, (;TP=0, FP=0, FN=0, precision=0.0, recall=0.0, f1=0.0, mse=0.0, mae=0.0))
        continue
    end
    
    # Evaluate
    eval_results = evaluate_model(model, x_eval, y_eval, threshold)

    # True labels & metrics
    n = length(values)
    n_prob = Int(floor(probation_ratio * n))
    ts_eval = ts[n_prob + window_size + 1 : end]

    true_flags = load_true_labels(label_path, filename, ts_eval)
    metrics = compute_metrics(filename, eval_results.anomaly_flags, true_flags)

    # Save model & metrics
    # save_model(model, filename)
    save_metrics(filename, metrics)
end

println("\nAll files processed successfully")