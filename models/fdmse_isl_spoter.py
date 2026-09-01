import torch
import torch.nn as nn


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    SPOTER-style Transformer decoder layer.

    The original SPOTER architecture removes decoder
    self-attention and retains cross-attention followed
    by the feed-forward network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # No decoder self-attention.
        x = tgt

        # Cross-attention.
        x = x + self._mha_block(
            self.norm2(x),
            memory,
            memory_mask,
            memory_key_padding_mask,
            memory_is_causal,
        )

        # Feed-forward block.
        x = x + self._ff_block(self.norm3(x))

        return x


class FDMSEISLSPOTER(nn.Module):
    """
    SPOTER adapted for FDMSE-ISL.

    Input:
        (T, 177)

        59 landmarks × 3 coordinates
        = 177 features/frame

    Architecture:
        177
          ↓
        Linear(177 → 108)
          ↓
        SPOTER Transformer
          ↓
        53 classes
    """

    def __init__(
        self,
        num_classes=53,
        input_dim=177,
        hidden_dim=108,
        nhead=9,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # -------------------------------------------------
        # FDMSE-ISL input projection
        # -------------------------------------------------

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # -------------------------------------------------
        # SPOTER learned positional representation
        # -------------------------------------------------

        self.row_embed = nn.Parameter(
            torch.rand(50, hidden_dim)
        )

        # The original SPOTER implementation uses the
        # first learned row embedding as the positional
        # vector and broadcasts it across the sequence.
        self.pos = nn.Parameter(
            torch.cat(
                [
                    self.row_embed[0].unsqueeze(0).repeat(
                        1, 1, 1
                    )
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(0)
        )

        # -------------------------------------------------
        # Learned class query
        # -------------------------------------------------

        self.class_query = nn.Parameter(
            torch.rand(1, hidden_dim)
        )

        # -------------------------------------------------
        # Transformer encoder
        # -------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # -------------------------------------------------
        # SPOTER decoder
        # -------------------------------------------------

        decoder_layer = SPOTERTransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------

        self.linear_class = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape: (T, 177)

        Returns
        -------
        logits : torch.Tensor
            Shape: (1, 53)
        """

        if inputs.ndim != 2:
            raise ValueError(
                f"Expected input shape (T, 177), "
                f"got {tuple(inputs.shape)}"
            )

        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features/frame, "
                f"got {inputs.shape[-1]}"
            )

        # -------------------------------------------------
        # Input projection
        # -------------------------------------------------

        x = self.input_projection(inputs.float())

        # (T, 108)

        # Add batch dimension.
        x = x.unsqueeze(1)

        # (T, 1, 108)

        # SPOTER positional representation.
        x = self.pos + x

        # -------------------------------------------------
        # Transformer encoder
        # -------------------------------------------------

        memory = self.encoder(x)

        # (T, 1, 108)

        # -------------------------------------------------
        # Class query
        # -------------------------------------------------

        query = self.class_query.unsqueeze(0)

        # (1, 1, 108)

        # -------------------------------------------------
        # Transformer decoder
        # -------------------------------------------------

        decoded = self.decoder(
            query,
            memory,
        )

        # (1, 1, 108)

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        logits = self.linear_class(
            decoded.transpose(0, 1)
        )

        # (1, 1, 53)

        logits = logits.squeeze(1)

        # (1, 53)

        return logits


if __name__ == "__main__":

    print("=" * 70)
    print("FDMSE-ISL SPOTER ARCHITECTURE TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "Capability:",
            torch.cuda.get_device_capability(0)
        )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = FDMSEISLSPOTER(
        num_classes=53,
        input_dim=177,
        hidden_dim=108,
    ).to(device)

    print("\nModel created.")

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # -----------------------------------------------------
    # Variable-length tests
    # -----------------------------------------------------

    print("\nVariable sequence-length tests:")

    for T in [90, 120, 150, 180, 270]:

        x = torch.randn(
            T,
            177,
            device=device
        )

        with torch.no_grad():
            output = model(x)

        print(
            f"T={T:3d} | "
            f"Input={tuple(x.shape)} | "
            f"Output={tuple(output.shape)}"
        )

    # -----------------------------------------------------
    # GPU synchronization
    # -----------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("ARCHITECTURE TEST PASSED")
    print("=" * 70)
