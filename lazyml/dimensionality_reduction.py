def reduce_dimensionality(name, X_train, y_train, X_test, **kwargs):

    if name == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(**kwargs)
        pca.fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    elif name == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if "n_components" in kwargs and int(kwargs["n_components"]) >= len(
            set(y_train)
        ):
            raise ValueError("LDA requires n_components < n_classes.")

        lda = LinearDiscriminantAnalysis(**kwargs)
        lda.fit(X_train, y_train)
        return lda.transform(X_train), lda.transform(X_test)
    else:
        raise NotImplementedError(f"{name} not implemented.")
