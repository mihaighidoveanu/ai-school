//
//  ReadersAndWriters_Fair.hpp
//  Lab
//
//  Created by Ciprian on 10/13/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef ReadersAndWriters_Fair_hpp
#define ReadersAndWriters_Fair_hpp

#include <mutex>
#include "MyFairSemaphore2.hpp"
#include "Utils.hpp"
#include "SimulationSettings.h"

class ReadersAndWriters_Fair
{
public:
    
    void Write(const int id)
    {
        
        // Wait to be serviced
        m_serviceQueue.wait(&m_allThreadsContext[id]);
        
        // Served ; Now try to enter in safe zone
        m_resourcesLock.wait(); // We have to wait every readers and writers in the system
        
        m_serviceQueue.signal(); // Let another one to be serviced; Might be still locking the resource !
        
        // We are in safe zone...do WRITING
        std::ostringstream msgin; msgin <<"ID " << id <<  " +++++ I'm WRITING.\n";
        Utils::writeConsoleMessageSynchronized(msgin);
        
        Utils::SleepRandomTime(100, 500);
        
        std::ostringstream msgout; msgout << "ID " << id << " ----- Finished WRITING\n";
        Utils::writeConsoleMessageSynchronized(msgout);
        //------------
        
        
        // Leaving safe zone
        m_resourcesLock.signal(); // Add a signal, maybe someone else will continue
    }
    
    void Read(const int id)
    {
        // Wait to be serviced
        m_serviceQueue.wait(&m_allThreadsContext[id]);
        
        // Serviced, now do normal reader flow to enter in safe zone
        m_readersLock.wait();
        m_numReaders++;
        if (m_numReaders == 1) // No readers, take the resource to readers
            m_resourcesLock.wait();
        m_readersLock.signal();
        
        // Let another entity to be serviced
        m_serviceQueue.signal();
        
        
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
            m_resourcesLock.signal(); // RElease the resource, give a chance to writer..if he is the next one.
        m_readersLock.signal();
    }
    
    ReadersAndWriters_Fair()
    {
        m_numReaders = 0;
        m_readersLock.init(1);
        m_resourcesLock.init(1);
        m_serviceQueue.init(1);
    }
    
private:
    
    MyFairThreadContext m_allThreadsContext[NUM_READERS + NUM_WRITERS];
    
    int m_numReaders;
    MySemaphore m_readersLock; // Controls access to readers counters and logic
    MySemaphore m_resourcesLock; // Control access between readers versus readers to the shared resource
    MyFairSemaphore2 m_serviceQueue; // Fair queue semaphore. Will serve items in the order of their appearance
};



#endif /* ReadersAndWriters_Fair_hpp */
