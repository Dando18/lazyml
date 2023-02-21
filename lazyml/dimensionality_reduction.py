def reduce_dimensionality(name, X_train, X_test, **kwargs):

    if name == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(**kwargs)
        pca.fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    else:
        raise NotImplementedError(f"{name} not implemented.")
