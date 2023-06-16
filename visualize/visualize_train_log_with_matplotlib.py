# 学習ログ解析


def evaluate_history_by_matplotlib(history):
    import japanize_matplotlib
    import matplotlib.pyplot as plt

    # 損失と精度の確認
    print(f"初期状態: 損失: {history[0,3]:.5f} 正解率: {history[0,4]:.5f}")
    print(f"最終状態: 損失: {history[-1,3]:.5f} 正解率: {history[-1,4]:.5f}")

    num_epochs = len(history)
    unit = num_epochs / 10

    plt.figure(figsize=(12, 7))

    # 学習曲線の表示 (損失)
    ax = plt.subplot(1, 2, 1)
    ax.plot(history[:, 0], history[:, 1], "b", label="訓練")
    ax.plot(history[:, 0], history[:, 3], "k", label="検証")
    ax.set_xticks(np.arange(0, num_epochs + 1, unit))
    ax.set_xlabel("繰り返し回数")
    ax.set_ylabel("損失")
    ax.set_title("学習曲線(損失)")
    ax.legend()

    # 学習曲線の表示 (精度)
    ax = plt.subplot(1, 2, 2)
    ax.plot(history[:, 0], history[:, 2], "b", label="訓練")
    ax.plot(history[:, 0], history[:, 4], "k", label="検証")
    ax.set_xticks(np.arange(0, num_epochs + 1, unit))
    ax.set_xlabel("繰り返し回数")
    ax.set_ylabel("正解率")
    ax.set_title("学習曲線(正解率)")
    ax.legend()

    # 表示
    plt.show()
