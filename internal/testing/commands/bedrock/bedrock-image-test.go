package bedrock

import (
	"context"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	llmproviders "github.com/manishiitg/multi-llm-provider-go"
	"github.com/manishiitg/multi-llm-provider-go/internal/recorder"
	"github.com/manishiitg/multi-llm-provider-go/internal/testing"
	"github.com/manishiitg/multi-llm-provider-go/internal/testing/commands/shared"
)

var BedrockImageTestCmd = &cobra.Command{
	Use:   "bedrock-image",
	Short: "Test Bedrock image understanding (vision)",
	Run:   runBedrockImageTest,
}

type bedrockImageTestFlags struct {
	model     string
	imagePath string
	imageURL  string
	record    bool
	replay    bool
	testDir   string
}

var bedrockImageFlags bedrockImageTestFlags

func init() {
	BedrockImageTestCmd.Flags().StringVar(&bedrockImageFlags.model, "model", "", "Bedrock model to test")
	BedrockImageTestCmd.Flags().StringVar(&bedrockImageFlags.imagePath, "image-path", "", "Path to image file (JPEG, PNG, GIF, WebP)")
	BedrockImageTestCmd.Flags().StringVar(&bedrockImageFlags.imageURL, "image-url", "", "URL of image to test")
	BedrockImageTestCmd.Flags().BoolVar(&bedrockImageFlags.record, "record", false, "Record LLM responses")
	BedrockImageTestCmd.Flags().BoolVar(&bedrockImageFlags.replay, "replay", false, "Replay recorded responses")
	BedrockImageTestCmd.Flags().StringVar(&bedrockImageFlags.testDir, "test-dir", "testdata", "Test directory")
}

func runBedrockImageTest(cmd *cobra.Command, args []string) {
	_ = godotenv.Load("agent_go/.env")
	_ = godotenv.Load(".env")
	_ = godotenv.Load("../.env")

	logFile := viper.GetString("log-file")
	logLevel := viper.GetString("log-level")
	testing.InitTestLogger(logFile, logLevel)
	logger := testing.GetTestLogger()

	// Get model ID
	modelID := bedrockImageFlags.model
	if modelID == "" {
		modelID = os.Getenv("BEDROCK_PRIMARY_MODEL")
		if modelID == "" {
			modelID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
		}
	}

	log.Printf("🚀 Testing Bedrock Image Understanding with %s", modelID)

	ctx := context.Background()
	var rec *recorder.Recorder

	// Setup recorder if recording or replaying
	if bedrockImageFlags.record || bedrockImageFlags.replay {
		recConfig := recorder.RecordingConfig{
			Enabled:  bedrockImageFlags.record,
			TestName: "image", // Test name for test suite
			Provider: "bedrock",
			ModelID:  modelID,
			BaseDir:  bedrockImageFlags.testDir,
		}
		rec = recorder.NewRecorder(recConfig)
		if bedrockImageFlags.replay {
			rec.SetReplayMode(true)
		}
		ctx = recorder.WithRecorder(ctx, rec)
	}

	// Create Bedrock LLM using our adapter
	llm, err := llmproviders.InitializeLLM(llmproviders.Config{
		Provider:    llmproviders.ProviderBedrock,
		ModelID:     modelID,
		Temperature: 0.7,
		Logger:      logger,
		Context:     ctx,
	})
	if err != nil {
		log.Printf("❌ Failed to create Bedrock LLM: %v", err)
		return
	}

	// Run shared image test with context
	if bedrockImageFlags.imagePath != "" || bedrockImageFlags.imageURL != "" {
		shared.RunImageTestWithContextAndImage(ctx, llm, modelID, bedrockImageFlags.imagePath, bedrockImageFlags.imageURL)
	} else {
		shared.RunImageTestWithContext(ctx, llm, modelID)
	}
}
