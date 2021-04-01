import neptune


def activate_monitoring(user, project):
    return neptune.init(project_qualified_name=f'{user}/{project}')


def create_exp(hyper_params, tags):
    neptune.create_experiment(
        name='sentiment-analysis',
        params=hyper_params,
        upload_source_files=['*.py', 'requirements.txt'],
        tags=tags,
        send_hardware_metrics=True
    )


def record_metadata(metrics):
    # log them in Neptune
    for metric, value in metrics.items():
        neptune.log_metric(metric, value)

    return metrics


def save_artifact(data_path, model_file):
    neptune.log_artifact(data_path)
    neptune.log_artifact(model_file)
    return None
