//
//  ReadersAndWriters_WritersPriority.hpp
//  Lab
//
//  Created by Ciprian on 10/13/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef ReadersAndWriters_WritersPriority_hpp
#define ReadersAndWriters_WritersPriority_hpp

#include <mutex>
#include "MySemphore.hpp"
#include "Utils.hpp"
#include "SimulationSettings.h"

class ReadersAndWriters_WriterPriority
{
public:
    
    void Write(const int id)
    {
        // Try to enter in safe zone
        m_writersLock.wait();
        m_numWriters++;
        
        // First entering writer blocks everybody else;
        // Notice that if there are multiple writers in the system, only when the last one exists will allow readers enter
        if (m_numWriters == 1)
            m_readersAttemptLock.wait();
        m_writersLock.signal();
        
        m_resourcesLock.wait(); // We have to wait every readers and writers in the system
        
        // We are in safe zone...do WRITING
        std::ostringstream msgin; msgin <<"ID " << id <<  " +++++ I'm WRITING.\n";
        Utils::writeConsoleMessageSynchronized(msgin);
        
        Utils::SleepRandomTime(100, 500);
        
        std::ostringstream msgout; msgout << "ID " << id << " ----- Finished WRITING\n";
        Utils::writeConsoleMessageSynchronized(msgout);
        m_resourcesLock.signal(); // Add a signal, maybe someone else will continue
        //------------
        
        // Leaving safe zone
        m_writersLock.wait();
        m_numWriters--;
        if (m_numWriters == 0)
            m_readersAttemptLock.signal();
        m_writersLock.signal();
    }
    
    // This solution uses another lock to wait on readers until no writer is requesting something at the moment
    void Read(const int id)
    {
        // Ensure mutual exclusion for readers. Semaphore used as classic mutex, modern way is to use lock guards as used in the semaphore
        
        // Try to enter in safe zone
        m_readersAttemptLock.wait(); // Check if we are allowed first
        m_readersLock.wait();
        m_numReaders++;
        if (m_numReaders == 1) // What would happen without the initial lock ?
        {
            // If first reader, block writers.
            m_resourcesLock.wait();
        }
        m_readersLock.signal(); // Release the semaphore entry
        m_readersAttemptLock.signal();
        
        // We are in safe zone...do READING
        std::ostringstream msgin; msgin << "ID: " << id << " +++++ I'm READING.\n";
        Utils::writeConsoleMessageSynchronized(msgin);
        Utils::SleepRandomTime(100, 500);
        std::ostringstream msgout; msgout << "ID: " << id << " ----- Finished READING\n";
        Utils::writeConsoleMessageSynchronized(msgout);
        //----
        
        // Exiting from safe zone
        m_readersLock.wait();
        m_numReaders--;
        if (m_numReaders == 0)
            m_resourcesLock.signal(); // RElease the resource,
        m_readersLock.signal();
    }
    
    ReadersAndWriters_WriterPriority()
    {
        m_numReaders = 0;
        m_numWriters = 0;
        m_writersLock.init(1);
        m_readersLock.init(1);
        m_readersAttemptLock.init(1);
        m_resourcesLock.init(1);
    }
    
private:
    int m_numReaders;
    int m_numWriters;
    MySemaphore m_readersLock; // Controls access to readers counters and logic
    MySemaphore m_writersLock;
    MySemaphore m_readersAttemptLock;
    MySemaphore m_resourcesLock; // Control access between readers versus readers to the shared resource
};

#endif /* ReadersAndWriters_WritersPriority_hpp */
