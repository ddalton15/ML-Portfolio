# visualize.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from graphviz import Digraph
from engine import Value
from nn import MLP

# ---- Dark theme for all plots ----
plt.style.use("dark_background")


# ========== 1. Animated Training Loss ==========


def make_dataset(n=100):
    """Generate a swirl/moon dataset."""
    np.random.seed(42)
    N = n // 2

    # Class +1: upper arc
    theta1 = np.linspace(0, np.pi, N)
    x1 = (
        np.column_stack([np.cos(theta1), np.sin(theta1)]) + np.random.randn(N, 2) * 0.15
    )

    # Class -1: lower arc shifted
    theta2 = np.linspace(0, np.pi, N)
    x2 = (
        np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
        + np.random.randn(N, 2) * 0.15
    )

    X = np.vstack([x1, x2]).tolist()
    y = [1.0] * N + [-1.0] * N
    return X, y


def train_with_history(model, X, y, steps=80, lr=0.05):
    """Train and record loss + predictions at each step."""
    history = {"loss": [], "preds": [], "params": []}

    for k in range(steps):
        # Forward
        ypred = [model(x)[0] for x in X]
        loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, y))

        # Zero grad
        for p in model.parameters():
            p.grad = 0.0

        # Backward (using your backward from engine.py [1])
        loss.backward()

        # Update
        for p in model.parameters():
            p.data -= lr * p.grad

        history["loss"].append(loss.data)
        history["preds"].append([p.data for p in ypred])
        history["params"].append([p.data for p in model.parameters()])

        if k % 10 == 0:
            acc = sum(
                (1 if p > 0 else -1) == int(yt)
                for p, yt in zip(history["preds"][-1], y)
            ) / len(y)
            print(f"  Step {k:3d} | Loss: {loss.data:8.4f} | Acc: {acc:.0%}")

    return history


# ========== 2. Decision Boundary ==========


def plot_decision_boundary(ax, model, X, y, resolution=50):
    """Draw a heatmap of model predictions over 2D space."""
    X_np = np.array(X)
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    Z = []
    for i in range(len(xx.ravel())):
        xi = [xx.ravel()[i], yy.ravel()[i]]
        out = model(xi)[0].data
        Z.append(out)
    Z = np.array(Z).reshape(xx.shape)

    cmap = ListedColormap(["#ff006620", "#00000000", "#00ff6620"])
    ax.contourf(xx, yy, Z, levels=[-1, -0.01, 0.01, 1], cmap=cmap)
    ax.contour(xx, yy, Z, levels=[0], colors=["#00ffcc"], linewidths=2)

    # Plot data points
    X_np = np.array(X)
    y_np = np.array(y)
    ax.scatter(
        X_np[y_np > 0, 0],
        X_np[y_np > 0, 1],
        color="#00ff99",
        edgecolors="white",
        s=40,
        zorder=5,
        label="+1",
    )
    ax.scatter(
        X_np[y_np < 0, 0],
        X_np[y_np < 0, 1],
        color="#ff4466",
        edgecolors="white",
        s=40,
        zorder=5,
        label="-1",
    )


# ========== 3. Gradient Flow Heatmap ==========


def plot_gradient_flow(ax, model):
    """Show gradient magnitudes across layers."""
    layers = model.layers
    grad_means = []
    grad_maxs = []
    labels = []

    for i, layer in enumerate(layers):
        grads = [abs(p.grad) for p in layer.parameters()]
        grad_means.append(np.mean(grads))
        grad_maxs.append(np.max(grads))
        labels.append(f"Layer {i}")

    x = range(len(labels))
    ax.bar(x, grad_maxs, color="#ff006680", label="Max |grad|", width=0.4, align="edge")
    ax.bar(
        x, grad_means, color="#00ffcc", label="Mean |grad|", width=-0.4, align="edge"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.set_title("⚡ Gradient Flow", fontsize=12, color="#00ffcc")


# ========== 4. Styled Computation Graph ==========


def draw_styled_graph(root, filename="styled_graph"):
    """Neon-styled computation graph based on your draw_dot [1]."""
    dot = Digraph(
        format="svg",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "#0d1117",
            "fontcolor": "white",
        },
    )

    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    for n in nodes:
        uid = str(id(n))
        label = getattr(n, "label", "")
        color = "#00ffcc" if n.grad != 0 else "#555555"
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (label, n.data, n.grad),
            shape="record",
            style="filled",
            fillcolor="#161b22",
            fontcolor=color,
            color=color,
        )
        if n._op:
            dot.node(
                name=uid + n._op,
                label=n._op,
                shape="circle",
                style="filled",
                fillcolor="#ff4466",
                fontcolor="white",
                color="#ff4466",
            )
            dot.edge(uid + n._op, uid, color="#ff446680")

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op, color="#00ffcc80")

    dot.render(filename, view=True)
    print(f"  ✅ Saved {filename}.svg")


# ========== 5. Main Dashboard ==========


def main():
    print("\n🧠 MICROGRAD VISUALIZER")
    print("=" * 50)

    # --- Dataset ---
    X, y = make_dataset(100)

    # --- Model ---
    np.random.seed(1337)
    model = MLP(2, [16, 16, 1])
    n_params = len(model.parameters())
    print(f"  Model: MLP(2, [16, 16, 1])")
    print(f"  Parameters: {n_params}")
    print("=" * 50)

    # --- Train ---
    print("\n📉 Training...")
    history = train_with_history(model, X, y, steps=80, lr=0.05)

    # --- Build Dashboard ---
    print("\n🎨 Generating dashboard...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "⚡ MICROGRAD TRAINING DASHBOARD ⚡",
        fontsize=18,
        color="#00ffcc",
        fontweight="bold",
    )

    # Plot 1: Loss curve with glow effect
    ax1 = fig.add_subplot(2, 2, 1)
    steps = range(len(history["loss"]))
    ax1.fill_between(steps, history["loss"], alpha=0.15, color="#00ffcc")
    ax1.plot(history["loss"], color="#00ffcc", linewidth=2)
    ax1.set_title("📉 Training Loss", color="#00ffcc")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.2)

    # Plot 2: Decision boundary
    ax2 = fig.add_subplot(2, 2, 2)
    plot_decision_boundary(ax2, model, X, y)
    ax2.set_title("🎯 Decision Boundary", color="#00ffcc")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # Plot 3: Gradient flow
    ax3 = fig.add_subplot(2, 2, 3)
    plot_gradient_flow(ax3, model)
    ax3.grid(alpha=0.2)

    # Plot 4: Parameter evolution
    ax4 = fig.add_subplot(2, 2, 4)
    param_history = np.array(history["params"])
    for i in range(min(20, param_history.shape[1])):
        ax4.plot(param_history[:, i], alpha=0.5, linewidth=0.8)
    ax4.set_title("🔀 Parameter Evolution (first 20)", color="#00ffcc")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Value")
    ax4.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("micrograd_dashboard.png", dpi=200, facecolor="#0d1117")
    plt.show()
    print("  ✅ Saved micrograd_dashboard.png")

    # --- Styled Computation Graph ---
    print("\n🔗 Generating computation graph...")
    x_sample = X[0]
    out = model(x_sample)[0]
    loss = (out - Value(y[0])) ** 2
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    draw_styled_graph(loss, "micrograd_graph")

    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()
