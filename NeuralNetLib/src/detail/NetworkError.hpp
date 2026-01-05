#pragma once

#include "pch.h"

namespace axon {

	enum class FileError : uint8_t
	{
		Ok,
		FileNotFound,
		CorruptFile,
		IncorrectFileType,
		CannotOpenFile
	};

	class FileErrorCategory : public std::error_category {

		const char* name() const noexcept override {
			return "FileError";
		}

		std::string message(int error_value) const override {
			switch (static_cast<FileError>(error_value))
			{
			case FileError::Ok:
				return "No error";
			case FileError::FileNotFound:
				return "Could not find file. Check the path to the file is correct.";
			case FileError::CorruptFile:
				return "The file you have tried to open has been corrupted.";
			case FileError::IncorrectFileType:
				return "The file you have opened is the incorrect file type.";
			case FileError::CannotOpenFile:
				return "Could not open file. This is usually because the program does not have the permissions to open the file.";
			default:
				return "Unknown error.";
			}
		}
	};

	inline const std::error_category& FileError_Category() {
		static FileErrorCategory instance;
		return instance;
	}

	inline std::error_code make_error_code(FileError error) {
		return { static_cast<int>(error), FileError_Category() };
	}

	enum class InterfaceError : uint8_t
	{
		Ok,
		InterfaceFailure,
		InterfaceUnavailable
	};

	class InterfaceErrorCategory : public std::error_category {
		const char* name() const noexcept override {
			return "InterfaceError";
		}

		std::string message(int error_value) const noexcept override {
			switch (static_cast<InterfaceError>(error_value)) {
			case InterfaceError::Ok:
				return "No error";
			case InterfaceError::InterfaceUnavailable:
				return "The requested interface either does not exist on your device or you have not installed the required libraries.";
			default:
				return "Unknown error.";
			}
		}
	};

	inline const std::error_category& InterfaceError_Category() {
		static InterfaceErrorCategory instance;
		return instance;
	}

	inline std::error_code make_error_code(InterfaceError error) {
		return { static_cast<int>(error), InterfaceError_Category() };
	}

	enum class UsageError : uint8_t
	{
		Ok,
		IncorrectInput,
		IncorrectState
	};

	class UsageErrorCategory : public std::error_category {
		const char* name() const noexcept override {
			return "UsageError";
		}

		std::string message(int error_value) const noexcept override {
			switch (static_cast<UsageError>(error_value)) {
			case UsageError::Ok:
				return "No error.";
			case UsageError::IncorrectInput:
				return "The input you have given to this function is incorrect, read the documentation for further information.";
			case UsageError::IncorrectState:
				return "Your network is in the incorrect state for this function, read the documentation for further information.";
			default:
				return "Unknown error.";
			}
		}
	};

	inline const std::error_category& UsageError_Category() {
		static UsageErrorCategory instance;
		return instance;
	}

	inline std::error_code make_error_code(UsageError error) {
		return { static_cast<int>(error), UsageError_Category() };
	}

}