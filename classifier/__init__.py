__all__ = ["M2IFEEnsemble", "MultiLabelClassifier"]


def __getattr__(name):
    if name == "M2IFEEnsemble":
        from .ensemble import M2IFEEnsemble

        return M2IFEEnsemble
    if name == "MultiLabelClassifier":
        from .model import MultiLabelClassifier

        return MultiLabelClassifier
    raise AttributeError(name)
