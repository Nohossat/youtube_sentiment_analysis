import neptune.new as neptune


def activate_monitoring(user, project):
    """
    Return a valid Neptune run instance
    :param user: name of the Neptune.ai user
    :param project: project to log metrics into
    :return: Neptune.new.run object
    """
    return neptune.init(project=f'{user}/{project}',
                        source_files=['*.py', 'requirements.txt'])


def create_exp(hyper_params, tags, run):
    """
    Additional values to add to the experiment
    :param hyper_params: Hyper parameters used for training
    :param tags: List of tags to link the project to in Neptune.ai
    :param run: Neptune.new.Run
    :return: None
    """
    run['parameters'] = hyper_params
    run['sys/tags'].add(tags)
    return None


def record_metadata(metrics, run):
    """
    Send data to Neptune.ai
    :param metrics: metrics to log
    :param run: Neptune.new.Run
    :return: None
    """
    # log them in Neptune
    for metric, value in metrics.items():
        run[metric] = value

    return None


def save_artifact(data_path, model_file, run):
    """

    :param data_path: filepath to the dataset
    :param model_file: filepath to the new model
    :param run: Neptune.new.Run
    :return: None
    """
    run["model"].upload(model_file)
    run["dataset"].upload(data_path)
    return None
