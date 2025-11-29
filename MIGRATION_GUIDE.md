# Migration Guide: Extracting llm-providers to Separate Repository

## ✅ Completed Steps

1. ✅ Updated `go.mod` module path to `github.com/manishiitg/multi-llm-provider-go`
2. ✅ Updated all internal imports in `llm-providers` directory
3. ✅ Updated `agent_go/go.mod` to use new module path
4. ✅ Updated `mcpagent/go.mod` to use new module path
5. ✅ Removed `llm-providers` from `go.work`
6. ✅ Updated README.md with new installation instructions

## 📋 Next Steps to Complete Migration

### Step 1: Copy Files to New Repository

```bash
# Navigate to your cloned repository
cd ~/multi-llm-provider-go

# Copy all files from llm-providers (excluding .git if it exists)
cp -r /Users/mipl/mcp-agent-builder-go/llm-providers/* .
cp -r /Users/mipl/mcp-agent-builder-go/llm-providers/.* . 2>/dev/null || true

# Remove any .git directory if copied
rm -rf .git
```

### Step 2: Initialize Git and Make Initial Commit

```bash
cd ~/multi-llm-provider-go

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Extract llm-providers as standalone module"

# Add remote (if not already added)
git remote add origin https://github.com/manishiitg/multi-llm-provider-go.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Create Initial Release Tag

```bash
cd ~/multi-llm-provider-go

# Create and push v0.1.0 tag
git tag v0.1.0
git push origin v0.1.0
```

### Step 4: Update Main Repository Dependencies

After pushing the new repository, update the main repository:

```bash
cd /Users/mipl/mcp-agent-builder-go

# Update agent_go dependencies
cd agent_go
go get github.com/manishiitg/multi-llm-provider-go@v0.1.0
go mod tidy

# Update mcpagent dependencies
cd ../mcpagent
go get github.com/manishiitg/multi-llm-provider-go@v0.1.0
go mod tidy

# Verify everything works
cd ..
go work sync
```

### Step 5: Test Everything

```bash
# Test the new repository independently
cd ~/multi-llm-provider-go
go mod tidy
go build ./...
go test ./...

# Test main repository with new dependency
cd /Users/mipl/mcp-agent-builder-go
go work sync
cd agent_go
go build ./...
cd ../mcpagent
go build ./...
```

### Step 6: Clean Up (Optional)

Once everything is verified working, you can optionally remove the old `llm-providers` directory from the main repository:

```bash
cd /Users/mipl/mcp-agent-builder-go
# Make sure everything works first, then:
# rm -rf llm-providers
```

## 🔍 Verification Checklist

- [ ] New repository is created and pushed to GitHub
- [ ] All files are copied to new repository
- [ ] Initial commit is made
- [ ] v0.1.0 tag is created and pushed
- [ ] `agent_go` can build successfully
- [ ] `mcpagent` can build successfully
- [ ] All tests pass in new repository
- [ ] All tests pass in main repository

## 📝 Notes

- The module path is now: `github.com/manishiitg/multi-llm-provider-go`
- All imports have been updated to use the new path
- The version is set to `v0.1.0` - update as needed
- Make sure to run `go mod tidy` in both repositories after migration

## 🚨 Important

Before removing the old `llm-providers` directory, ensure:
1. The new repository is working correctly
2. All dependent modules can build and run
3. All tests pass
4. You have a backup or the repository is pushed to GitHub

