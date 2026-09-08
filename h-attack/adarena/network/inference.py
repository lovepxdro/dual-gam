from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from dataclasses import dataclass

from typing import Sequence

import numpy as np

from .base import (
    CaptureBatch,
    FlowExtractor,
    FlowFeatureBatch,
)


@dataclass(frozen=True, slots=True)
class FeatureCompatibilityReport:
    """
    Resultado da verificação entre o schema esperado
    pelo modelo e o produzido pelo FlowExtractor.
    """

    expected_features: tuple[
        str,
        ...
    ]

    observed_features: tuple[
        str,
        ...
    ]

    unsupported_features: tuple[
        str,
        ...
    ]

    missing_features: tuple[
        str,
        ...
    ]

    extra_features: tuple[
        str,
        ...
    ]

    order_matches: bool

    finite: bool

    @property
    def ready_for_model(
        self,
    ) -> bool:

        return bool(
            not self.unsupported_features
            and not self.missing_features
            and not self.extra_features
            and self.order_matches
            and self.finite
        )


class IncompatibleFeatureSchemaError(
    ValueError
):
    """
    O dado extraído da rede não possui representação
    compatível com o modelo treinado.
    """

    def __init__(
        self,
        report: FeatureCompatibilityReport,
    ) -> None:

        self.report = report

        problems = []

        if report.unsupported_features:
            problems.append(
                "não reconstruídas: "
                + ", ".join(
                    report
                    .unsupported_features
                )
            )

        if report.missing_features:
            problems.append(
                "ausentes: "
                + ", ".join(
                    report
                    .missing_features
                )
            )

        if report.extra_features:
            problems.append(
                "extras: "
                + ", ".join(
                    report
                    .extra_features
                )
            )

        if not report.order_matches:
            problems.append(
                "ordem de features incompatível"
            )

        if not report.finite:
            problems.append(
                "valores não finitos"
            )

        message = (
            "Schema incompatível com o modelo"
        )

        if problems:
            message += (
                ": "
                + "; ".join(
                    problems
                )
            )

        super().__init__(
            message
        )


class FeatureSchemaValidator:
    """
    Garante que a saída do extractor pode entrar
    exatamente no preprocessador/modelo treinado.

    Não realiza imputação, reordenação silenciosa
    nem preenchimento de features faltantes.
    """

    def validate(
        self,
        *,
        expected_features: Sequence[str],
        batch: FlowFeatureBatch,
    ) -> FeatureCompatibilityReport:

        expected = tuple(
            str(feature)
            for feature
            in expected_features
        )

        observed = tuple(
            str(feature)
            for feature
            in batch.feature_names
        )

        expected_set = set(
            expected
        )

        observed_set = set(
            observed
        )

        missing = tuple(
            feature
            for feature in expected
            if feature
            not in observed_set
        )

        extra = tuple(
            feature
            for feature in observed
            if feature
            not in expected_set
        )

        return FeatureCompatibilityReport(
            expected_features=expected,

            observed_features=observed,

            unsupported_features=tuple(
                batch.unsupported_features
            ),

            missing_features=missing,

            extra_features=extra,

            order_matches=(
                expected
                == observed
            ),

            finite=bool(
                np.isfinite(
                    batch.X
                ).all()
            ),
        )

    def require_compatible(
        self,
        *,
        expected_features: Sequence[str],
        batch: FlowFeatureBatch,
    ) -> FeatureCompatibilityReport:

        report = self.validate(
            expected_features=(
                expected_features
            ),
            batch=batch,
        )

        if not report.ready_for_model:
            raise (
                IncompatibleFeatureSchemaError(
                    report
                )
            )

        return report


class BinaryPredictor(ABC):
    """
    Contrato mínimo entre a ADArena e um classificador
    binário utilizado durante inferência.

    O pipeline não precisa conhecer PyTorch.
    """

    @abstractmethod
    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class TorchBinaryPredictor(
    BinaryPredictor
):
    """
    Adapter de inferência para os defensores PyTorch
    utilizados atualmente pela ADArena.
    """

    def __init__(
        self,
        model,
        *,
        device: str = "cpu",
    ) -> None:

        self.model = model
        self.device = device

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        import torch

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "X deve possuir shape [N, features]"
            )

        tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():
            output = self.model(
                tensor
            )

        probabilities = (
            output
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(
                np.float32
            )
        )

        if (
            len(probabilities)
            != len(X)
        ):
            raise RuntimeError(
                "Defensor retornou quantidade "
                "inesperada de predições"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise RuntimeError(
                "Defensor retornou "
                "probabilidades não finitas"
            )

        if (
            (
                probabilities
                < 0
            ).any()
            or (
                probabilities
                > 1
            ).any()
        ):
            raise RuntimeError(
                "BinaryPredictor espera "
                "probabilidades entre 0 e 1"
            )

        return probabilities


@dataclass(slots=True)
class NetworkInferenceResult:
    """
    Resultado da observação de tráfego pelo Defensor.
    """

    extracted: FlowFeatureBatch

    compatibility: (
        FeatureCompatibilityReport
    )

    normalized_features: np.ndarray

    probabilities: np.ndarray

    predictions: np.ndarray

    threshold: float

    @property
    def n_flows(
        self,
    ) -> int:
        return int(
            len(
                self.predictions
            )
        )

    @property
    def attack_count(
        self,
    ) -> int:
        return int(
            (
                self.predictions
                == 1
            ).sum()
        )

    @property
    def benign_count(
        self,
    ) -> int:
        return int(
            (
                self.predictions
                == 0
            ).sum()
        )


class NetworkInferencePipeline:
    """
    Caminho rede -> Defensor.

    Fluxo:

        CaptureBatch
            ↓
        FlowExtractor
            ↓
        FeatureSchemaValidator
            ↓
        Preprocessor.normalizar()
            ↓
        BinaryPredictor
            ↓
        classificação

    Nenhuma inferência ocorre quando o schema não é
    exatamente compatível.
    """

    def __init__(
        self,
        *,
        extractor: FlowExtractor,
        preprocessor,
        predictor: BinaryPredictor,
        threshold: float = 0.5,
        validator: (
            FeatureSchemaValidator
            | None
        ) = None,
    ) -> None:

        if not (
            0.0
            <= threshold
            <= 1.0
        ):
            raise ValueError(
                "threshold deve estar "
                "entre 0 e 1"
            )

        self.extractor = extractor

        self.preprocessor = (
            preprocessor
        )

        self.predictor = predictor

        self.threshold = float(
            threshold
        )

        self.validator = (
            validator
            or FeatureSchemaValidator()
        )

    def inspect(
        self,
        capture: CaptureBatch,
    ) -> tuple[
        FlowFeatureBatch,
        FeatureCompatibilityReport,
    ]:
        """
        Extrai os fluxos e produz um diagnóstico
        de compatibilidade sem chamar o Defensor.
        """

        expected_features = tuple(
            self.preprocessor
            .feature_names
        )

        if not expected_features:
            raise ValueError(
                "Preprocessador não possui "
                "feature_names"
            )

        extracted = (
            self.extractor.extract(
                capture,

                feature_names=(
                    expected_features
                ),

                # Precisamos receber o diagnóstico
                # completo em vez de abortar dentro
                # do extractor.
                strict=False,
            )
        )

        compatibility = (
            self.validator.validate(
                expected_features=(
                    expected_features
                ),

                batch=extracted,
            )
        )

        return (
            extracted,
            compatibility,
        )

    def infer(
        self,
        capture: CaptureBatch,
    ) -> NetworkInferenceResult:

        (
            extracted,
            compatibility,
        ) = self.inspect(
            capture
        )

        if not (
            compatibility
            .ready_for_model
        ):
            raise (
                IncompatibleFeatureSchemaError(
                    compatibility
                )
            )

        normalized = (
            self.preprocessor
            .normalizar(
                extracted.X
            )
        )

        probabilities = (
            self.predictor
            .predict_proba(
                normalized
            )
        )

        if (
            len(probabilities)
            != extracted.X.shape[0]
        ):
            raise RuntimeError(
                "Quantidade de probabilidades "
                "incompatível com os fluxos "
                "extraídos"
            )

        predictions = (
            probabilities
            >= self.threshold
        ).astype(
            np.int64
        )

        return NetworkInferenceResult(
            extracted=extracted,

            compatibility=(
                compatibility
            ),

            normalized_features=(
                normalized
            ),

            probabilities=(
                probabilities
            ),

            predictions=(
                predictions
            ),

            threshold=self.threshold,
        )
