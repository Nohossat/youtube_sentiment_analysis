import neptune.new as neptune


def activate_monitoring(user, project):
    return neptune.init(project=f'{user}/{project}',
                        source_files=['*.py', 'requirements.txt'])


def create_exp(hyper_params, tags, run):
    run['parameters'] = hyper_params
    run['sys/tags'].add(tags)

    """
    neptune.create_experiment(
        name='sentiment-analysis',
        params=hyper_params,
        upload_source_files=['*.py', 'requirements.txt'],
        tags=tags,
        send_hardware_metrics=True
    )
    """
    return None


def record_metadata(metrics, run):
    # log them in Neptune
    for metric, value in metrics.items():
        run[metric] = value

    return None


def save_artifact(data_path, model_file, run):
    run["model"].upload(model_file)
    run["dataset"].upload(data_path)
    return None
