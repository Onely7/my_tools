# 学習ログ解析

def evaluate_history_by_plotly(history):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 正解率: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 正解率: {history[-1,4]:.5f}')

    num_epochs = len(history)
    unit = num_epochs / 10


    plot_fig = make_subplots(rows=1, cols=2, subplot_titles=['学習曲線(損失)', '学習曲線(正解率)'], horizontal_spacing=0.15, vertical_spacing=0.2)

    # 学習曲線の表示 (損失)
    cmap = plt.get_cmap('cividis')  # カラーマップの取得
    c = cmap(np.array(range(0, 255, 255//2)))
    plot_fig.add_trace(go.Scatter(x=history[:,0], y=history[:,1], name='訓練', line={"width": 2, "color": f"rgb({c[1][0]}, {c[1][1]}, {c[1][2]})"}), row=1, col=1)
    plot_fig.add_trace(go.Scatter(x=history[:,0], y=history[:,3], name='検証', line={"width": 2, "color": f"rgb({c[0][0]}, {c[0][1]}, {c[0][2]})"}), row=1, col=1)
    plot_fig.update_xaxes(range=[0, num_epochs+1], title='繰り返し回数', row=1, col=1)
    plot_fig.update_yaxes(title='損失', row=1, col=1)

    # 学習曲線の表示 (精度)
    cmap = plt.get_cmap('twilight')  # カラーマップの取得
    c = cmap(np.array(range(0, 255, 255//2)))
    plot_fig.add_trace(go.Scatter(x=history[:,0], y=history[:,2], name='訓練', line={"width": 2, "color": f"rgb({c[1][0]}, {c[1][1]}, {c[1][2]})"}), row=1, col=2)
    plot_fig.add_trace(go.Scatter(x=history[:,0], y=history[:,4], name='検証', line={"width": 2, "color": f"rgb({c[2][0]}, {c[2][1]}, {c[2][2]})"}), row=1, col=2)
    plot_fig.update_xaxes(range=[0, num_epochs+1], title='繰り返し回数', row=1, col=2)
    plot_fig.update_yaxes(title='正解率', row=1, col=2)

    # 表示
    plot_fig.show()
